
function irtrapweights(pts)
  out = zeros(length(pts))
  out[1]   = 0.5*(pts[2]-pts[1])
  out[end] = 0.5*(pts[end] - pts[end-1])
  for j in 2:(length(out)-1)
    out[j] = 0.5*(pts[j+1] - pts[j]) + 0.5*(pts[j] - pts[j-1])
  end
  out
end

function simulate_process(pts, kernel, m; rng=Random.default_rng())
  K = [kernel(x, y) for x in pts, y in pts]
  L = cholesky!(Symmetric(K)).L
  L*randn(rng, length(pts), m)
end

gen_wgrid(pts::Vector{Float64}, Ω) = range(-Ω, Ω, length=length(pts))

# TODO (cg 2025/04/12 14:34): For now, this only gives you squares in
# higher-dimensional frequency space. That can certainly be lifted, but will
# keep it simple for now.
function gen_wgrid(pts::Vector{SVector{D,Float64}}, Ω::Float64) where{D}
  len1d   = Int(ceil(length(pts)^(1/D)))
  wgrid1d = range(-Ω, Ω, length=len1d)
  vec(SVector{D,Float64}.(Iterators.product(fill(wgrid1d, D)...)))
end

static_points(x::Vector{Float64}) = [SA[xj] for xj in x]
static_points(x::Vector{SVector{D,Float64}}) where{D} = x

struct SincKernel{D} <: Function
  Ωv::NTuple{D,Float64}
  perturb::Float64
end

function (sk::SincKernel{D})(x::SVector{D,Float64}, y::SVector{D,Float64}) where{D}
  xmy  = x-y
  out  = prod(1:D) do j
    Ωj = sk.Ωv[j]
    2*Ωj*sinc(2*Ωj*xmy[j])
  end
  iszero(xmy) && (out += sk.perturb)
  out
end

getdim(pts::Vector{SVector{D,Float64}}) where{D} = 1

# TODO (cg 2025/04/21 12:26): At some point, it would be nice to split each of
# these things out into weakdeps so that the dependencies of this package can be
# smaller. But with that said, if users will almost always want to pull in the
# Krylov solver (for example), maybe it should just be a hard dependency.
abstract type LinearSystemSolver end

# For IrregularSpectraLowRankApproxExt:
struct SketchSolver <: LinearSystemSolver 
  tol::Float64
end

struct DenseSolver <: LinearSystemSolver  end

function solve_linsys(pts, win, Ω, solver::DenseSolver; verbose=false)
  wgrid = range(-Ω, Ω, length=length(pts))
  b     = linsys_rhs(win, wgrid)
  F     = nudftmatrix(pts, wgrid, +1)
  qr!(F, ColumnNorm())\b
end

abstract type KrylovPreconditioner end

struct HMatrixPreconditioner <: KrylovPreconditioner
  atol::Float64   # 1e-8
  luatol::Float64 # 1e-8
end

struct CholeskyPreconditioner <: KrylovPreconditioner end

struct KrylovSolver{P} <: LinearSystemSolver where{P}
  preconditioner::P
  λ::Float64
end

function krylov_preconditioner(pts_sa, Ω, solver::KrylovSolver{CholeskyPreconditioner};
                               verbose=false)
  kern = SincKernel(ntuple(j->Ω, getdim(pts_sa)), solver.λ)
  M = threaded_km_assembly(kern, pts_sa)
  pre_time = @elapsed Mf = cholesky!(M)
  verbose && @printf "Preconditioner assembly time: %1.3fs\n" pre_time
  Mf
end

function solve_linsys(pts, win, Ω, solver::KrylovSolver; verbose=false)
  # TODO (cg 2025/03/28 18:13): think harder about what this nquad should be.
  # Yes it is cheap to crank it up, but if this can be reduced then naturally
  # it should be.
  (wgrid, glwts) = glquadrule(krylov_nquad(pts, win), -Ω, Ω)
  rhs      = linsys_rhs(win, wgrid)
  pts_sa   = static_points(pts)
  D        = Diagonal(sqrt.(glwts))
  F        = NUFFT3(pts, collect(wgrid.*(2*pi)), false, 1e-15, D)
  pre      = krylov_preconditioner(pts_sa, Ω, solver; verbose=verbose)
  vrb = verbose ? 10 : 0
  lsmr(F, D*rhs, N=pre, verbose=vrb, etol=0.0, axtol=0.0, atol=1e-11, 
       btol=0.0, rtol=1e-10, conlim=Inf, ldiv=true, itmax=500)[1]

end

function default_solver(pts)
  if length(pts) > 5_000
    @warn """For large datasets, the default solver (Krylov with a dense Cholesky preconditioner)
    can have a long runtime. Consider ]add-ing the weakdep HMatrices and using the following
    solver instead:

    ```
      using HMatrices
      pre = HMatrixPreconditioner(1e-8, 1e-8)
      solver = KrylovSolver(pre, 1e-8)

      estimate_sdf([...], solver=solver, [...])
    ```

    """
  end
  KrylovSolver(CholeskyPreconditioner(), 1e-8)
end

function solve_linsys(pts, win, Ω, solver=default_solver(pts); verbose=false)
  solve_linsys(pts, win, Ω, solver; verbose=verbose)
end

# generic broadcasted Fourier transform.
fouriertransform(g, wv::AbstractVector) = fouriertransform.(Ref(g), wv)

chebnodes(n) = reverse([cos(pi*(2*k-1)/(2*n)) for k in 1:n])./2 .+ 0.5
chebnodes(n, a, b) = chebnodes(n)*(b-a) .+ a

function glquadrule(n::Int64, a, b)
  (no, wt) = gausslegendre(n)
  (bmad2, bpad2) = ((b-a)/2, (b+a)/2)
  @inbounds for j in 1:n
    no[j] = no[j]*bmad2 + bpad2
    wt[j] = wt[j]*bmad2
  end
  (no, wt)
end

function glquadrule(nv::NTuple{N, Int64}, a::NTuple{N,Float64},
                    b::NTuple{N,Float64}) where{N}
  no_wt_v = [glquadrule(nv[j], a[j], b[j]) for j in 1:N]
  nodes   = vec(SVector{N,Float64}.(Iterators.product(getindex.(no_wt_v, 1)...)))
  weights = vec(prod.(Iterators.product(getindex.(no_wt_v, 2)...)))
  (nodes, weights)
end

function glquadrule(nv::NTuple{N,Int64}, a::Float64, b::Float64) where{N}
  glquadrule(nv, ntuple(j->a, N), ntuple(j->b, N))
end

function segment_glquadrule(intervals, m; add=0)
  nodes_weights = map(intervals) do (aj, bj)
    glquadrule(Int(ceil(m*(bj - aj))) + add, aj, bj)
  end
  (reduce(vcat, getindex.(nodes_weights, 1)), 
   reduce(vcat, getindex.(nodes_weights, 2)))
end

segment_glquadrule_nyquist(intervals, Ω) = segment_glquadrule(intervals, Ω*4, add=50)

function threaded_km_assembly(kernel::K, pts) where{K}
  out = Matrix{Float64}(undef, length(pts), length(pts))
  Threads.@threads for k in 1:length(pts)
    for j in 1:k
      out[j,k] = kernel(pts[j], pts[k])
    end
  end
  Symmetric(out, :U)
end

function has_empty_leaves(H)
  sparse_leaves = filter(x->HMatrices.isadmissible(x), HMatrices.leaves(H))
  (rmin, rmax)  = extrema(x->HMatrices.rank(x.data), sparse_leaves)
  iszero(rmin) 
end

