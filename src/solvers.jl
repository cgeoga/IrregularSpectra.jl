
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

#
# SketchSolver: using the LowRankApprox.jl-powered extensions, you can utilize
# low-rank structure in the normal equation system to accelerate computations.
# this is a direct method. For certain kernels in certain dimensions, however,
# you might expect the rank of the normal equation matrix to be about n/2, so
# you'll still die in O(n^3) land. And so you should either use a GaussKernel or
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
#
# is defined in an extension.
struct KrylovSolver{P,K} <: LinearSystemSolver where{P,K}
  preconditioner::P
  pre_kernel::Type{K}
  perturbation::Float64
  maxit::Int64
end

abstract type KrylovPreconditioner end

function KrylovSolver(p, pre_kernel::Type{K}=DefaultKernel; 
                      perturbation=1e-8, maxit=500) where{K}
  KrylovSolver(p, pre_kernel, perturbation, maxit)
end
                      
struct HMatrixPreconditioner <: KrylovPreconditioner
  tol::Float64  # generic suggestion: 1e-8
  ftol::Float64 # generic suggestion: 1e-8
end

struct VecchiaPreconditioner <: KrylovPreconditioner
  ncond::Int64 # generic suggestion: 50
  nfsa::Int64  # generic suggestion: 30
end

struct CholeskyPreconditioner <: KrylovPreconditioner end


function krylov_preconditioner!(pts_sa, Ω, solver::KrylovSolver{CholeskyPreconditioner,K};
    verbose=false) where{K}
  kernel = gen_kernel(solver, pts_sa, Ω)
  M = threaded_km_assembly(kernel, pts_sa)
  pre_time = @elapsed Mf = cholesky!(M)
  verbose && @printf "Preconditioner assembly time: %1.3fs\n" pre_time
  (true, Mf)
end

function solve_linsys(pts, win, Ω, solver::KrylovSolver; verbose=false)
  (wgrid, glwts) = glquadrule(krylov_nquad(pts, win), .-Ω, Ω)
  rhs            = linsys_rhs(win, wgrid)
  pts_sa         = static_points(pts)
  wgrid_sa       = static_points(wgrid)
  kernel         = gen_kernel(solver, pts_sa, Ω)
  D              = Diagonal(sqrt.(glwts.*fouriertransform.(Ref(kernel), wgrid)))
  (_ldiv, pre)   = krylov_preconditioner!(pts_sa, Ω, solver; verbose=verbose)
  F              = NUFFT3(pts_sa, collect(wgrid_sa.*(2*pi)), false, 1e-15, D)
  vrb            = verbose ? 10 : 0
  wts = lsmr(F, D*rhs, N=pre, verbose=vrb, etol=0.0, axtol=0.0, atol=1e-11, 
             btol=0.0, rtol=1e-10, conlim=Inf, ldiv=_ldiv, itmax=solver.maxit)[1]
  l2norm = let tmp = Vector{ComplexF64}(undef, length(rhs))
    _F = NUFFT3(pts_sa, collect(wgrid_sa.*(2*pi)), false, 1e-15)
    mul!(tmp, _F, wts)
    sqrt(dot(glwts, abs2.(tmp)))
  end
  wts ./= l2norm
  wts
end

function default_solver(pts; perturbation=1e-10)
  if length(pts) < 2000
    return DenseSolver()
  else
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
    return KrylovSolver(CholeskyPreconditioner(), DefaultKernel; 
                        perturbation=perturbation)
  end
end

