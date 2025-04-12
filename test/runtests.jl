
using Test, LinearAlgebra, StableRNGs, IrregularSpectra
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

@testset "Matern selector" begin
  let scope_dummy = 0
    include("./tests/matern_selector.jl")
  end
end

@testset "Fast sinc" begin
  let scope_dummy = 0
    include("./tests/fastsinc.jl")
  end
end

@testset "Prolate" begin
  let scope_dummy = 0
    include("./tests/prolate.jl")
  end
end

