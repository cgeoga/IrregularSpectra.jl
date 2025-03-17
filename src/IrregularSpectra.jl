module IrregularSpectra

  using Random, Printf, Statistics, LinearAlgebra, SparseArrays, QuadGK, LowRankApprox, FINUFFT, Bessels

  export window_quadrature_weights, estimate_sdf, Kaiser, matern_frequency_selector

  include("utils.jl")

  include("nufft.jl")

  include("window.jl")

  include("transform.jl")

  include("matern_selector.jl")

end 
