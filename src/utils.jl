
# A dense NUDFT matrix. 
function nudftmatrix(s1, s2, sgn; T=Float64)
  sgn in (-1.0, 1.0) || throw(error("sign should be -1.0 or 1.0! You provided $sgn."))
  [cispi(2*sgn*T(dot(sj, sk))) for sj in s1, sk in s2]
end

# v ↦ D*(F*v)
struct PreNUFFT3
  F::NUFFT3
  D::Diagonal{Float64, Vector{Float64}}
  adj_buf::Vector{ComplexF64}
end

Base.size(pnf::PreNUFFT3)    = size(pnf.F)
Base.size(pnf::PreNUFFT3, j) = size(pnf.F, j)
Base.eltype(pnf::PreNUFFT3)  = ComplexF64
LinearAlgebra.adjoint(pnf::PreNUFFT3) = Adjoint{ComplexF64, PreNUFFT3}(pnf)

function LinearAlgebra.mul!(buf::Vector{ComplexF64}, pnf::PreNUFFT3, 
                            v::Vector{ComplexF64})
  mul!(buf, pnf.F, v)
  for j in eachindex(buf)
    buf[j] *= pnf.D[j,j]
  end
  buf
end

# TODO (cg 2025/06/25 13:58): this allocates, b
function LinearAlgebra.mul!(buf::Vector{ComplexF64}, 
                            apnf::Adjoint{ComplexF64, PreNUFFT3}, 
                            v::Vector{ComplexF64})
  mul!(apnf.parent.adj_buf, apnf.parent.D, v)
  mul!(buf, apnf.parent.F', apnf.parent.adj_buf)
end

function PreNUFFT3(s1::Vector{SVector{S,Float64}}, s2::Vector{SVector{S,Float64}},
                   sgn::Int, D) where{S}
  F    = NUFFT3(s1, s2, sgn)
  abuf = Vector{ComplexF64}(undef, length(s1))
  PreNUFFT3(F, D, abuf)
end

max_col_norm(M) = maximum(norm, eachcol(M))

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

function simple_adaptive_integrate(fn::F, a, b; init_size=128, 
                                   max_size=2^15, ctol=1e-10) where{F}
  n       = init_size
  rule_n  = glquadrule(n, a, b)
  rule_2n = glquadrule(2*n, a, b)
  int_n   = dot(fn.(rule_n[1]),  rule_n[2])
  int_2n  = dot(fn.(rule_2n[1]), rule_2n[2])
  isconv  = abs(int_n - int_2n)/abs(int_2n) < ctol
  while !isconv
    @info "doubling...$(2*n)"
    n      *=  2
    n > max_size && throw(error("Could not reach convergence tolerance $ctol with <$max_size nodes."))
    rule_n  = rule_2n
    rule_2n = glquadrule(2*n, a, b)
    int_n   = int_2n
    int_2n  = dot(fn.(rule_2n[1]), rule_2n[2])
    isconv  = abs(int_n - int_2n)/abs(int_2n) < ctol
  end
  int_2n
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

function y_slice_intervals(xval, ygrid, missing_sorted)
  y    = Set(copy(ygrid))
  gixs = searchsorted(missing_sorted, xval, by=t->t[1])
  gaps = sort(unique(getindex.(missing_sorted[gixs], 2)))
  foreach(v->delete!(y, v), gaps)
  gappy_intervals(sort(collect(y)); minlen=0)
end

function gappy_rule(xgrid::AbstractVector{Float64}, 
                    ygrid::AbstractVector{Float64},
                    missing_sorted::Vector{SVector{2,Float64}}, 
                    rule_density)
  # First, let get a quadrature rule along the x-axis.
  (nox, wtx) = IrregularSpectra.segment_glquadrule([extrema(xgrid)], rule_density[1])
  # Now, for each nox[j], we'll find the nearest row of in xgrid, get the
  # gappy interval decomposition of that marginal slice, and then obtain a
  # quadrature rule on that gappy 1D grid. This can (and should!) get improved
  # to handle more complex regions more thoughtfully. But for now, we'll just
  # babysit the results a little and see how it goes.
  row_rules = map(eachindex(nox, wtx)) do j
    (noxj, wtxj) = (nox[j], wtx[j])
    ix = searchsortedfirst(xgrid, noxj)
    abs(xgrid[ix-1] - noxj) < abs(xgrid[ix] - noxj) && (ix -= 1)
    ivs = y_slice_intervals(xgrid[ix], ygrid, missing_sorted)
    (no, wt) = IrregularSpectra.segment_glquadrule(ivs, rule_density[2])
    wt .*= wtxj
    nodes_2d = [SA[noxj, nok] for nok in no]
    (nodes_2d, wt)
  end
  (reduce(vcat, getindex.(row_rules, 1)), reduce(vcat, getindex.(row_rules, 2)))
end

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

# TODO (cg 2025/05/16 15:39): Weirdly, ldiv!(ch::CHOLMOD.Factor[...], rhs) or
# whatever doesn't exist. So we need this little wrapper. I think this is
# fixed in 1.12, but probably good to support LTS.
struct LDivWrapper{T}
  M::T
end

Base.eltype(ldw::LDivWrapper{T}) where{T}  = eltype(ldw.M)
Base.size(ldw::LDivWrapper{T}) where{T}    = size(ldw.M)
Base.size(ldw::LDivWrapper{T}, j) where{T} = size(ldw.M, j)

function LinearAlgebra.ldiv!(ldw::LDivWrapper{T}, v::Vector{ComplexF64}) where{T}
  tmp = ldw.M\v
  v  .= tmp
end

function LinearAlgebra.ldiv!(buf::Vector{ComplexF64}, ldw::LDivWrapper{T}, v::Vector{ComplexF64}) where{T}
  buf .= v
  ldiv!(ldw, buf)
end

function inrange1d(pts::Vector{Float64}, x::Float64, radius::Float64)
  i1 = max(searchsortedfirst(pts, x-radius), 1)
  i2 = min(searchsortedfirst(pts, x+radius), length(pts))
  i1:i2
end

function gss(obj, _a, _b; maxit=1000, atol=1e-8)
  (a,b) = (_a, _b)
  c = b - (b-a)/MathConstants.golden
  d = a + (b-a)/MathConstants.golden
  (oc, od) = (obj(c), obj(d))
  for _ in 1:maxit
    if oc < od
      b=d
      d=c
      c= b - (b-a)/MathConstants.golden
      od=oc
      oc=obj(c)
    else
      a=c
      c=d
      d=a + (b-a)/MathConstants.golden
      oc=od
      od=obj(d)
    end
    if abs(b-a)<atol
      (isequal(a, _a) || isequal(b, _b)) && throw(error("No solution found in window ($_a, $_b)"))
      return (b+a)/2
    end
  end
  throw(error("No convergence to tolerance $atol in $maxit iterations."))
end

function gappy_grid_Ω(pts::Vector{Float64}; info=true)
  issorted(pts) || throw(error("You should be providing this function (and all functions in this package) sorted points if you are in 1D."))
  dpts     = sort(diff(pts))
  min_diff = dpts[1]
  quarter_quantile_diff = dpts[div(length(dpts), 4)]
  is_gridded = any(j->isapprox(dpts[j], quarter_quantile_diff, rtol=1e-2), 1:10)
  (is_gridded && info) && @info "Points appear to be on a gappy lattice, picking grid-based Nyquist frequency Ω. If that is not correct, please supply your own Ω."
  (is_gridded, inv(min_diff)/2)
end

function gappy_grid_Ω(pts::Vector{SVector{D,Float64}}; info=true) where{D}
  ptsj    = [sort(unique(getindex.(pts, j))) for j in 1:D]
  results = gappy_grid_Ω.(ptsj; info=false)
  is_gridded = all(x->x[1], results)
  (is_gridded && info) && @info "Points appear to be on a gappy lattice, picking grid-based Nyquist frequency Ω. If that is not correct, please supply your own Ω."
  (is_gridded, ntuple(j->results[j][2], D))
end

