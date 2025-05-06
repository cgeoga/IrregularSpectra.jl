
module IrregularSpectra

  # stdlibs:
  using Random, Printf, Statistics, LinearAlgebra, SparseArrays

  # external pacakges (actual dependencies):
  using QuadGK, FINUFFT, Bessels, FastGaussQuadrature, Krylov, StaticArrays

  include("utils.jl")

  include("nufft.jl")

  include("solvers.jl")
  export DenseSolver, SketchSolver, KrylovSolver, CholeskyPreconditioner, HMatrixPreconditioner, VecchiaPreconditioner

  include("kernels.jl")
  export SincKernel, GaussKernel

  include("window.jl")
  export Kaiser, Sine, Prolate1D, Prolate2D, TensorProduct2DWindow

  include("transform.jl")
  export window_quadrature_weights, estimate_sdf

end 

