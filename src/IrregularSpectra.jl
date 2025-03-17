module IrregularSpectra

  using Printf, LinearAlgebra, QuadGK, LowRankApprox, FINUFFT, Bessels

  export window_quadrature_weights, estimate_sdf, Kaiser

  include("utils.jl")

  include("nufft.jl")

  include("window.jl")

  include("transform.jl")

  include("matern_selector.jl")

end 
