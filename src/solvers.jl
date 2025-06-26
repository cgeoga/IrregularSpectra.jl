
abstract type LinearSystemSolver end

#
# DenseSolver: a full pivoted QR factorization. Arguably the most stable and
# definitive answer, but obviously also the slowest.
#
struct DenseSolver <: LinearSystemSolver  end

function solve_linsys(pts, win, Ω, solver::DenseSolver; verbose=false)
  wgrid = glquadrule(krylov_nquad(pts, win), .-Ω, Ω)[1]
  b     = linsys_rhs(win, wgrid)
  F     = nudftmatrix(wgrid, pts, -1)
  qr!(F, ColumnNorm())\b
end

#= DEPRECATED: there is just no circumstance where this solver is the best
#              choice at this point, and moving to pre-planned NUFFTs that make
#              the KrylovSolver faster make this even slower. So I'm just going
#              to remove it.
#
# SketchSolver: using the LowRankApprox.jl-powered extensions, you can utilize
# low-rank structure in the normal equation system to accelerate computations.
# this is a direct method. For certain kernels in certain dimensions, however,
# you might expect the rank of the normal equation matrix to be about n/2, so
# you'll still die in O(n^3) land. And so you should either use a KaiserKernel or
# instead just use the KrylovSolver below.
#
# NOTE: the method 
#
# solve_linsys(pts, win, Ω, solver::SketchSolver; verbose)
#
# is defined in extension(s).
struct SketchSolver{K} <: LinearSystemSolver 
  kernel::Type{K}
  sketchtol::Float64
  regularizer::Float64
end

SketchSolver(sketchtol::Float64) = SketchSolver(IdentityKernel, sketchtol, 1e-12)
=#


#
# KrylovSolver: the default and most sophisticated option, which uses an
# iterative solver with a cleverly chosen preconditioner to rapidly obtain
# estimator weights. Below you will see several different varieties of
# preconditioners that are implemented, some of which require the HMatrices
# extension. For small data sizes (n ≤ 10k, say), the CholeskyPreconditioner is
# probably going to be the fastest. 
#
# NOTE: the methods
#
# solve_linsys(pts, win, Ω, solver::KrylovSolver{HMatrixPreconditioner}; verbose)
# solve_linsys(pts, win, Ω, solver::KrylovSolver{VecchiaPreconditioner}; verbose)
# solve_linsys(pts, win, Ω, solver::KrylovSolver{SparsePreconditioner};  verbose)
#
# are defined in an extension.
struct KrylovSolver{P,K} <: LinearSystemSolver where{P,K}
  preconditioner::P
  pre_kernel::Type{K}
  perturbation::Float64
  maxit::Int64
end

abstract type KrylovPreconditioner end

struct NoPreconditioner <: KrylovPreconditioner end
default_perturb(pre::NoPreconditioner)  = 0.0
                      
struct HMatrixPreconditioner <: KrylovPreconditioner
  tol::Float64  # generic suggestion: 1e-8
  ftol::Float64 # generic suggestion: 1e-8
end
default_perturb(pre::HMatrixPreconditioner)  = 1e-6

struct VecchiaPreconditioner <: KrylovPreconditioner
  ncond::Int64 # generic suggestion: 30
end
default_perturb(pre::VecchiaPreconditioner)  = 1e-3

struct CholeskyPreconditioner <: KrylovPreconditioner end
default_perturb(pre::CholeskyPreconditioner) = 1e-10

struct SparsePreconditioner <: KrylovPreconditioner
  drop_tol::Float64
end
default_perturb(pre::SparsePreconditioner) = 1e-10

function KrylovSolver(p; pre_kernel::Type{K}=DefaultKernel, maxit=500,
                      perturbation=default_perturb(p)) where{K}
  KrylovSolver(p, pre_kernel, perturbation, maxit)
end

function krylov_preconditioner!(pts_sa, Ω, solver::KrylovSolver{NoPreconditioner,K};
                                verbose=false) where{K}
  (false, I)
end

function krylov_preconditioner!(pts_sa, Ω, solver::KrylovSolver{CholeskyPreconditioner,K};
                                verbose=false) where{K}
  kernel = gen_kernel(solver, pts_sa, Ω)
  M = threaded_km_assembly(kernel, pts_sa)
  pre_time = @elapsed Mf = cholesky!(M)
  verbose && @printf "Preconditioner assembly time: %1.3fs\n" pre_time
  (true, Mf)
end

function krylov_preconditioner!(pts_sa::Vector{SVector{1,Float64}}, Ω,
                                solver::KrylovSolver{SparsePreconditioner, K}; verbose=false) where{K}
  _pts   = getindex.(pts_sa, 1)
  issorted(_pts) || throw(error("For 1D sparse preconditioner, please pre-sort your points.")) 
  kernel = gen_kernel(solver, pts_sa, Ω)
  radius = kernel_tol_radius(Val(1), kernel, solver.preconditioner.drop_tol)
  pre_time = @elapsed begin
    ixs = [inrange1d(_pts, x[], radius) for x in pts_sa]
    I   = reduce(vcat, ixs)
    J   = reduce(vcat, [fill(j, length(ixs[j])) for j in eachindex(ixs)])
    V   = [kernel(_pts[jk[1]], _pts[jk[2]]) for jk in zip(I, J)]
    S   = Symmetric(sparse(I, J, V))
    Mf  = LDivWrapper(ldlt(S))
  end
  verbose && @printf "preconditioner assembly time: %1.3fs\n" pre_time
  (true, Mf)
end

function solve_linsys(pts, win, Ω, solver::KrylovSolver; verbose=false)
  (wgrid, glwts) = glquadrule(krylov_nquad(pts, win), .-Ω, Ω)
  rhs            = linsys_rhs(win, wgrid)
  pts_sa         = static_points(pts)
  wgrid_sa       = static_points(wgrid)
  kernel         = gen_kernel(solver, pts_sa, Ω)
  D              = Diagonal(sqrt.(glwts.*fouriertransform(kernel, wgrid)))
  (_ldiv, pre)   = krylov_preconditioner!(pts_sa, Ω, solver; verbose=verbose)
  F              = PreNUFFT3(collect(wgrid_sa.*(2*pi)), pts_sa, -1, D)
  vrb            = verbose ? 10 : 0
  wts = map(eachcol(rhs)) do rhsj
    lsmr(F, D*rhsj, N=pre, verbose=vrb, etol=0.0, axtol=0.0, atol=1e-11, 
         btol=0.0, rtol=1e-10, conlim=Inf, ldiv=_ldiv, itmax=solver.maxit)[1]
  end
  for wtsj in wts
    l2norm = let tmp = Vector{ComplexF64}(undef, size(rhs, 1))
      _F = NUFFT3(collect(wgrid_sa.*(2*pi)), pts_sa, -1)
      mul!(tmp, _F, wtsj)
      sqrt(dot(glwts, abs2.(tmp)))
    end
    wtsj ./= l2norm
  end
  reduce(hcat, wts)
end

function default_solver(pts; perturbation=1e-10)
  if length(pts) > 5_000
    @warn """For large datasets, the default solver (Krylov with a dense Cholesky preconditioner)
    can have a long runtime. Consider ]add-ing the weakdep HMatrices and using the following
    solver instead:

    ```
      using HMatrices
      solver = KrylovSolver(HMatrixPreconditioner(1e-8, 1e-8)) 
      estimate_sdf([...], solver=solver, [...])
    ```

    """
  end
  KrylovSolver(CholeskyPreconditioner(); perturbation=perturbation)
end

# Method added in an extension.
"""
`gridded_nyquist_gpss(pts::Vector{Float64}, bandwidth::Float64)`
`gridded_nyquist_gpss(pts::Vector{SVector{2,Float64}}, bandwidth::Float64)`

This function computes the well-concentrated GPSS functions of bandwidth
`bandwidth` (which is interpreted as a _radius_ in 2D).

**NOTE**: this function requires the extension for `ArnoldiMethod.jl`, so you
must `]add ArnoldiMethod` and `using ArnoldiMethod` in your file for this
function to have any methods.
"""
function gridded_nyquist_gpss end

