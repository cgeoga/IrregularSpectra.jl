
using Test, LinearAlgebra, StableRNGs, IrregularSpectra

@testset "recovery" begin
  let scope_dummy = 0
    include("./tests/recovery.jl")
  end
end

@testset "Matern selector" begin
  let scope_dummy = 0
    include("./tests/matern_selector.jl")
  end
end

@testset "Prolate" begin
  let scope_dummy = 0
    include("./tests/prolate.jl")
  end
end

