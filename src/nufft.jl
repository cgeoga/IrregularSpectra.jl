
# A dense NUDFT matrix. 
function nudftmatrix(s1, s2, sgn; T=Float64)
  sgn in (-1.0, 1.0) || throw(error("sign should be -1.0 or 1.0! You provided $sgn."))
  [cispi(2*sgn*T(dot(sj, sk))) for sj in s1, sk in s2]
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
#
# D is an optional addition so that this operator represents the action
# v ↦ D*(F*v).
struct NUFFT3{T}
  s1::Vector{Float64}
  s2::Vector{Float64}
  sgn::Bool
  tol::Float64
  D::T
end

NUFFT3(s1, s2, sgn, tol) = NUFFT3(collect(s1), collect(s2), sgn, tol, I)

Base.eltype(nf::NUFFT3)       = ComplexF64
Base.size(nf::NUFFT3)         = (length(nf.s2), length(nf.s1))
Base.size(nf::NUFFT3, j::Int) = size(nf)[j]

LinearAlgebra.adjoint(nf::NUFFT3) = Adjoint{ComplexF64, NUFFT3}(nf)

LinearAlgebra.ishermitian(nf::NUFFT3)  = false
function LinearAlgebra.mul!(buf, nf::NUFFT3, x)
  ifl = nf.sgn ? Int32(1) : Int32(-1)
  nufft1d3!(nf.s1, collect(x), ifl, nf.tol, nf.s2, buf)
  buf .= nf.D*buf
end

function LinearAlgebra.mul!(buf, anf::Adjoint{ComplexF64, NUFFT3}, x)
  nf  = anf.parent
  ifl = nf.sgn ? Int32(-1) : Int32(1)
  nufft1d3!(nf.s2, collect(nf.D'*x), ifl, nf.tol, nf.s1, buf)
  buf
end

# As currently parameterized, this is only for symmetric sinc matrices. But
# certainly this code could be trivially extended to the nonsymmetric case.
#
# TODO (cg 2025/03/23 15:55): could store buffers here to make mul! more efficient.
struct FastSinc{T}
  D::T
  s_2pi::Vector{Float64}
  no::Vector{Float64}
  wt::Vector{Float64}
end

# bandwidth bw parameterized here so that bw=1.0 corresponds to Julia's sinc
# function.
function FastSinc(s, bw=1.0; D=I)
  issymmetric(D) || throw(error("For now, the conjugation matrix D must be symmetric."))
  quadn    = Int(ceil(2*bw*maximum(abs, s)*4 + 50))
  (no, wt) = glquadrule(quadn, a=-0.5, b=0.5)
  FastSinc(D, s.*(2*pi*bw), no, wt)
end

LinearAlgebra.ishermitian(fs::FastSinc) = true
Base.eltype(fs::FastSinc)  = Float64 
Base.size(fs::FastSinc)    = (length(fs.s_2pi), length(fs.s_2pi))
Base.size(fs::FastSinc, j) = size(fs)[j]

# TODO (cg 2025/03/23 15:20): optimize this routine.
function LinearAlgebra.mul!(buf, fs::FastSinc, v) 
  nft_data   = nufft1d3(fs.s_2pi, complex.(fs.D*v), 1, 1e-15, fs.no)
  nft_data .*= fs.wt
  buf       .= fs.D*real(nufft1d3(fs.no, nft_data, -1, 1e-15, fs.s_2pi))
end

