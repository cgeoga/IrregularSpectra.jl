
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

function simulate_process(pts, kernel)
  K = [kernel(x, y) for x in pts, y in pts]
  L = cholesky!(Symmetric(K)).L
  L*randn(length(pts))
end

