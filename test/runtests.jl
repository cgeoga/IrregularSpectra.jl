
using Test, StableRNGs, IrregularSpectra

@testset "recovery" begin
  include("./tests/recovery.jl")
end

@testset "Matern selector" begin
  include("./tests/matern_selector.jl")
end

