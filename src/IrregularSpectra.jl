
module IrregularSpectra

  # stdlibs:
  using Random, Printf, Statistics, LinearAlgebra, SparseArrays

  # external pacakges (actual dependencies):
  using QuadGK, FINUFFT, Bessels, FastGaussQuadrature, Krylov, StaticArrays, BandlimitedOperators

  include("utils.jl")

  include("intervals.jl")
  export gappy_intervals

  include("solvers.jl")
  export DenseSolver, KrylovSolver, CholeskyPreconditioner, HMatrixPreconditioner, SparsePreconditioner

  include("window.jl")
  export Kaiser, Sine, Prolate1D, default_prolate_bandwidth, Prolate2D, TensorProduct2DWindow

  include("kernels.jl")

  include("transform.jl")
  export window_quadrature_weights, estimate_sdf

end 

