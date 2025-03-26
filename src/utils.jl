
function irtrapweights(pts)
  out = zeros(length(pts))
  out[1]   = 0.5*(pts[2]-pts[1])
  out[end] = 0.5*(pts[end] - pts[end-1])
  for j in 2:(length(out)-1)
    out[j] = 0.5*(pts[j+1] - pts[j]) + 0.5*(pts[j] - pts[j-1])
  end
  out
end

maximum_neighbor_dist(pts::Vector{Float64}) = maximum(diff(sort(pts)))

function simulate_process(pts, kernel, m; rng=Random.default_rng())
  K = [kernel(x, y) for x in pts, y in pts]
  L = cholesky!(Symmetric(K)).L
  L*randn(rng, length(pts), m)
end

function solve_linsys(pts, win, Ω; method, tol=1e-14, verbose=false)
  if method == :sketch
    wgrid = range(-Ω, Ω, length=length(pts))
    b     = linsys_rhs(win, wgrid)
    F     = NUFFT3(pts, wgrid.*(2*pi), false, 1e-15)
    Fo    = LinearOperator(F)
    Fqr   = pqrfact(Fo; rtol=tol)
    verbose && @printf "Rank of reduced QR: %i\n" size(Fqr.Q, 2)
    return Fqr\b
  elseif method == :krylov
    (wgrid, glwts) = IrregularSpectra.glquadrule(length(pts) + 100, a=-Ω, b=Ω)
    rhs      = IrregularSpectra.linsys_rhs(win, wgrid)
    pts_sa   = [SA[x] for x in pts]
    D        = Diagonal(sqrt.(glwts))
    F        = IrregularSpectra.NUFFT3(pts, wgrid.*(2*pi), false, 1e-15, D)
    kern     = (x,y) -> 2*Ω*sinc(2*Ω*(x[]-y[])) + Float64(x[]==y[])*1e-8
    sk       = KernelMatrix(kern, pts_sa, pts_sa)
    pre_time = @elapsed begin
      H  = assemble_hmatrix(sk; atol=1e-8)
      Hf = has_empty_leaves(H) ? I : lu(H; atol=1e-8)
    end
    verbose && @printf "Preconditioner assembly time: %1.3fs\n" pre_time
    vrb = verbose ? 10 : 0
    return lsmr(F, D*rhs, N=Hf, verbose=vrb, etol=1e-13, ldiv=true, itmax=500)[1]
  elseif method == :dense
    wgrid = range(-Ω, Ω, length=length(pts))
    b     = linsys_rhs(win, wgrid)
    F     = nudftmatrix(pts, wgrid, +1)
    return qr!(F, ColumnNorm())\b
  else
    throw(error("The two presently implemented methods are method=:sketch or method=:dense."))
  end
end

# generic broadcasted Fourier transform.
fouriertransform(g, wv::AbstractVector) = fouriertransform.(Ref(g), wv)

chebnodes(n) = reverse([cos(pi*(2*k-1)/(2*n)) for k in 1:n])./2 .+ 0.5
chebnodes(n, a, b) = chebnodes(n)*(b-a) .+ a

function glquadrule(n::Int64; a=-1.0, b=1.0)
  (no, wt) = gausslegendre(n)
  (bmad2, bpad2) = ((b-a)/2, (b+a)/2)
  @inbounds for j in 1:n
    no[j] = no[j]*bmad2 + bpad2
    wt[j] = wt[j]*bmad2
  end
  (no, wt)
end

function segment_glquadrule_nyquist(intervals, Ω)
  nodes_weights = map(intervals) do (aj, bj)
    glquadrule(Int(ceil(Ω*4*(bj - aj) + 50)), a=aj, b=bj)
  end
  (reduce(vcat, getindex.(nodes_weights, 1)), 
   reduce(vcat, getindex.(nodes_weights, 2)),
   nodes_weights)
end

function has_empty_leaves(H)
  sparse_leaves = filter(x->HMatrices.isadmissible(x), HMatrices.leaves(H))
  (rmin, rmax)  = extrema(x->HMatrices.rank(x.data), sparse_leaves)
  iszero(rmin) 
end
