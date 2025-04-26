module IrregularSpectra

  # stdlibs:
  using Random, Printf, Statistics, LinearAlgebra, SparseArrays

  # external pacakges (actual dependencies):
  using QuadGK, FINUFFT, Bessels, FastGaussQuadrature, Krylov, StaticArrays

  export DenseSolver, SketchSolver, KrylovSolver, CholeskyPreconditioner, HMatrixPreconditioner
  export window_quadrature_weights, estimate_sdf, Kaiser, Prolate1D, Sine

  include("utils.jl")

  include("nufft.jl")

  include("window.jl")

  include("transform.jl")

  include("matern_selector.jl")

end 
