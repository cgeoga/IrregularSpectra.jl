
module IrregularSpectra

  # stdlibs:
  using Random, Printf, Statistics, LinearAlgebra, SparseArrays

  # external pacakges (actual dependencies):
  using QuadGK, FINUFFT, Bessels, FastGaussQuadrature, Krylov, StaticArrays

  include("utils.jl")

  include("solvers.jl")
  export DenseSolver, SketchSolver, KrylovSolver, CholeskyPreconditioner, HMatrixPreconditioner

  include("nufft.jl")

  include("window.jl")
  export Kaiser, Prolate1D, Sine, TensorProduct2DWindow

  include("kernels.jl")
  export SincKernel, GaussKernel

  include("transform.jl")
  export window_quadrature_weights, estimate_sdf

end 

