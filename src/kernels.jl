

# Some object k::K<:KernelFunction must have the methods:
#
# k(pt1::SVector{D,Float64}, pt2::SVector{D,Float64}) -> F64
# fouriertransform(k, freq::SVector{D,Float64}) -> F64
# gen_kernel(ks::KrylovSolver{P,K}, pts::Vector{SVector{D,Float64}}, Ω) where{P} -> k.
# gen_kernel(ks::SketchSolver{K}, pts::Vector{SVector{D,Float64}}, Ω) where{P} -> k.
#
# That last method is because the actual spec of KrylovSolver doesn't do any
# actual computing, and it doesn't contain enough information to fully specify
# the kernel (in particular, the dimension of the process). So what gets stored
# in there is actually the _type_ K, and then given full information deeper in
# the call stack the object k::K is built.
abstract type KernelFunction end

#
# IdentityKernel: a special type to indicate no quadrature system at all. The
# idea here is that you have a fully special and specific dispatch on this to
# just do a straight QR/sketch/etc method without using any quadrature rule so
# that the matrices in your normal equations resemble kernel matrices.
#

struct IdentityKernel <: KernelFunction end


#
# Sinckernel: works well in 1D and involves no implicit regularization of higher
# frequencies.
#

struct SincKernel{D} <: KernelFunction
  Ωv::NTuple{D,Float64}
  perturbation::Float64
end

SincKernel(Ω::Float64, perturbation::Float64) = SincKernel((Ω,), perturbation)

function (sk::SincKernel{D})(x::SVector{D,Float64}, y::SVector{D,Float64}) where{D}
  xmy  = x-y
  out  = prod(1:D) do j
    Ωj = sk.Ωv[j]
    2*Ωj*sinc(2*Ωj*xmy[j])
  end
  iszero(xmy) && (out += sk.perturbation)
  out
end

function fouriertransform(sk::SincKernel{D}, w::SVector{D,Float64}) where{D} 
  Float64(all(j->abs(w[j]) <= sk.Ωv[j], 1:D))
end

fouriertransform(sk::SincKernel{1}, w::Float64) = fouriertransform(sk, SA[w])

function gen_kernel(ks::KrylovSolver{P,SincKernel}, 
                    pts::Vector{SVector{D,Float64}}, Ω) where{P,D}
  SincKernel(Ω, ks.perturbation)
end

function gen_kernel(ks::SketchSolver{SincKernel}, 
                    pts::Vector{SVector{D,Float64}}, Ω) where{D}
  SincKernel(Ω, 0.0)
end



#
# GaussKernel: works well in _any_ dimension because it is not oscillatory. It
# involves heavy implicit regularization of higher frequencies, so the
# transformed linear system will often give back weights whose spectral window
# is _better_ than what you asked for in the RHS of the original system.
#

struct GaussKernel{D} <: KernelFunction
  Mv::NTuple{D,Float64}
  prodmv::Float64
  perturbation::Float64
end

function GaussKernel(Mv::NTuple{D,Float64}; perturbation=1e-8) where{D}
  pmv = prod(Mv)
  GaussKernel(Mv, pmv, perturbation)
end

function (gk::GaussKernel{D})(x::SVector{D,Float64}, y::SVector{D,Float64}) where{D}
  inner = sum(j->abs2(gk.Mv[j]*(x[j] - y[j])), 1:D)
  out   = exp(-pi*inner)
  x == y ? out + gk.perturbation : out
end

function fouriertransform(gk::GaussKernel{D}, w::SVector{D,Float64}) where{D}
  inner = sum(j->abs2(w[j]/gk.Mv[j]), 1:D)
  exp(-pi*inner)/gk.prodmv
end

fouriertransform(gk::GaussKernel{1}, w::Float64) = fouriertransform(gk, SA[w])

function gen_kernel(ks::KrylovSolver{P,GaussKernel},
                    pts::Vector{SVector{D,Float64}}, Ω) where{P,D}
  GaussKernel(Ω; perturbation=ks.perturbation)
end

function gen_kernel(ks::SketchSolver{GaussKernel},
                    pts::Vector{SVector{D,Float64}}, Ω) where{D}
  GaussKernel(Ω; perturbation=0.0)
end

