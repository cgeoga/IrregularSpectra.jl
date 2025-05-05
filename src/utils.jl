
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

getdim(pts::Vector{Float64}) = 1
getdim(pts::Vector{SVector{D,Float64}}) where{D} = D

# default method:
krylov_nquad(pts::Vector{Float64}, win) = 4*length(pts) + 100
function krylov_nquad(pts::Vector{SVector{D,Float64}}, win) where{D}
  ntuple(_->Int(ceil(sqrt(4*length(pts)))) + 10, D)
end

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

function matern_cov(t, p)
  (sig, rho, v) = p
  iszero(t) && return sig^2
  arg = sqrt(2*v)*norm(t)/rho
  (sig^2)*((2^(1-v))/Bessels.gamma(v))*(arg^v)*Bessels.besselk(v, arg)
end

function matern_sdf(w, p) 
  d = length(w)
  (sig, rho, v) = p
  fpi2 = 4*pi^2
  pre  = (2^d)*(pi^(d/2))*Bessels.gamma(v+d/2)*((2*v)^v)/(Bessels.gamma(v)*rho^(2*v))
  (sig^2)*pre*(2*v/(rho^2) + fpi2*norm(w)^2)^(-v - d/2)
end

# This is a very quick-and-dirty function to compute a sparse inverse Cholesky
# factor (or rather, an LDLt factorization with Sigma ≈ inv(L'*inv(D)*L)).  It
# is normally the backbone of Vecchia-type methods, but in this setting it is
# basically exact.
#
# TODO (cg 2024/11/21 11:41): this is my max-simple demonstration of the rchol.
# This could of course be optimized a great deal. Including:
# -- using banded matrices instead of sparse
# -- pre-allocating buffers and parallelizing 
# -- I'm sure other stuff too
# But I doubt that this will _ever_ be the bottleneck even as it is, so it
# doesn't seem like a high priority to me.
function sparse_rchol(kfn, pts; k=20)
  n = length(pts)
  L = sparse(Diagonal(ones(n)))
  D = zeros(n)
  for j in 1:n
    cix  = max(1, j-k):(j-1)
    pt   = pts[j]
    cpts = pts[cix]
    Sjj  = kfn(pt, pt)
    if j > 1
      Scc = cholesky!(Hermitian([kfn(x,y) for x in cpts, y in cpts]))
      Scj = [kfn(x, pt) for x in cpts]
      kweights = Scc\Scj
      ccov     = Sjj - dot(Scj, kweights)
      D[j]     = ccov
      view(L, j, cix) .= -kweights
    else
      D[j] = Sjj
    end
  end
  (LowerTriangular(L), Diagonal(D))
end

