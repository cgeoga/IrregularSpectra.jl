
module IrregularSpectra

  # stdlibs:
  using Random, Printf, Statistics, LinearAlgebra, SparseArrays

  # external pacakges (actual dependencies):
  using Bessels, Krylov, StaticArrays, BandlimitedOperators
  import BandlimitedOperators: glquadrule

  include("utils.jl")

  include("intervals.jl")
  export gappy_intervals

  include("solvers.jl")
  export DenseSolver, KrylovSolver, NoPreconditioner, DensePreconditioner, VecchiaPreconditioner, HMatrixPreconditioner, SparsePreconditioner, gridded_nyquist_gpss

  include("window.jl")
  export Kaiser, Sine, Prolate1D, default_prolate_bandwidth, Prolate2D

  include("kernels.jl")

  include("transform.jl")
  export window_quadrature_weights, estimate_sdf

end 

