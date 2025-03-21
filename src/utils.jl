
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

function solve_linsys(pts, wgrid, b; method, tol=1e-14)
  if method == :sketch
    F   = NUFFT3(pts, wgrid.*(2*pi), true, 1e-15)
    Fo  = LinearOperator(F)
    Fqr = pqrfact(Fo; rtol=tol)
    return Fqr\b
  elseif method == :dense
    F   = nudftmatrix(pts, wgrid, +1)
    return qr!(F, ColumnNorm())\b
  else
    throw(error("The two presently implemented methods are method=:sketch or method=:dense."))
  end
end

# generic broadcasted Fourier transform.
fouriertransform(g, wv::AbstractVector) = fouriertransform.(Ref(g), wv)

