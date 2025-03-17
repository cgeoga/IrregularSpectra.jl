
# A dense NUDFT matrix. 
function nudftmatrix(s1, s2, sgn; T=Float64)
  sgn in (-1.0, 1.0) || throw(error("sign should be -1.0 or 1.0! You provided $sgn."))
  [cispi(2*T(dot(sj, sk))) for sj in s1, sk in s2]
end

# A matrix-free but O(n*m) NUDFT application. 
function nft(pts, fgrid, v; sgn::Int)
  sgn in (-1, 1) || throw(error("Sign arg needs to be in (-1, 1)."))
  (n1, n2) = (length(pts), length(fgrid))
  out      = zeros(ComplexF64, length(fgrid))
  sign2    = 2*sgn
  @inbounds begin
    Threads.@threads for j in eachindex(fgrid)
      fj = fgrid[j]
      @simd for k in eachindex(pts)
        out[j] += cispi(sign2*dot(pts[k], fgrid[j]))*v[k]
      end
    end
  end
  out
end

# TODO (cg 2024/10/03 14:57): could easily generalize this to be 2D or 3D with a
# parametric type of s{1,2}::Vector{T} where T and then specialize the `mul!`s
# as well.
struct NUFFT3
  s1::Vector{Float64}
  s2::Vector{Float64}
  sgn::Bool
  tol::Float64
end

Base.eltype(nf::NUFFT3)       = ComplexF64
Base.size(nf::NUFFT3)         = (length(nf.s2), length(nf.s1))
Base.size(nf::NUFFT3, j::Int) = size(nf)[j]

LinearAlgebra.adjoint(nf::NUFFT3) = Adjoint{ComplexF64, NUFFT3}(nf)

LinearAlgebra.ishermitian(nf::NUFFT3)  = false
function LinearAlgebra.mul!(buf, nf::NUFFT3, x)
  ifl = nf.sgn ? Int32(1) : Int32(-1)
  nufft1d3!(nf.s1, collect(x), ifl, nf.tol, nf.s2, buf)
  buf
end

function LinearAlgebra.mul!(buf, anf::Adjoint{ComplexF64, NUFFT3}, x)
  nf  = anf.parent
  ifl = nf.sgn ? Int32(-1) : Int32(1)
  nufft1d3!(nf.s2, collect(x), ifl, nf.tol, nf.s1, buf)
  buf
end

