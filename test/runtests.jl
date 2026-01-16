
using Test, StableRNGs, IrregularSpectra
using HMatrices # extensions
using IrregularSpectra.StaticArrays
using IrregularSpectra.LinearAlgebra

@testset "recovery" begin
  let scope_dummy = 0
    include("./tests/recovery.jl")
  end
end

@testset "Prolate" begin
  let scope_dummy = 0
    include("./tests/prolate.jl")
  end
end

