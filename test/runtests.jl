
using Test, LinearAlgebra, StableRNGs, IrregularSpectra
using LowRankApprox, HMatrices # extensions
using IrregularSpectra.FINUFFT
using IrregularSpectra.StaticArrays

@testset "NUFFT" begin
  let scope_dummy = 0
    include("./tests/nufft.jl")
  end
end

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

