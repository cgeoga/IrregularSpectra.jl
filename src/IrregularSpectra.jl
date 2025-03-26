module IrregularSpectra

  # stdlibs:
  using Random, Printf, Statistics, LinearAlgebra, SparseArrays

  # external pacakges (actual dependencies):
  using QuadGK, LowRankApprox, FINUFFT, Bessels, FastGaussQuadrature, ArnoldiMethod, Krylov, StaticArrays, HMatrices

  export window_quadrature_weights, estimate_sdf, Kaiser, matern_frequency_selector, Prolate1D

  include("utils.jl")

  include("nufft.jl")

  include("window.jl")

  include("transform.jl")

  include("matern_selector.jl")

end 
