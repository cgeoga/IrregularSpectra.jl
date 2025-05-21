

# Some object k::K<:KernelFunction must have the methods:
#
# k(pt1::SVector{D,Float64}, pt2::SVector{D,Float64}) -> F64
# fouriertransform(k, freqs::Vector{SVector{D,Float64}}) -> F64
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
# struct DefaultKernel: a special type that abstracts away the preconditioner
# choice from users. In 1D, it is the SincKernel, and in 2+D it is the
# KaiserKernel.
#
struct DefaultKernel end

function gen_kernel(ks::KrylovSolver{P,DefaultKernel},
                    pts::Vector{SVector{D,Float64}}, 
                    Ω) where{P,D}
  _k  = D == 1 ? SincKernel : KaiserKernel
  _ks = KrylovSolver(ks.preconditioner, _k, ks.perturbation, ks.maxit)
  gen_kernel(_ks, pts, Ω)
end

function gen_kernel(ks::SketchSolver{DefaultKernel},
                    pts::Vector{SVector{D,Float64}}, 
                    Ω) where{D}
  _k  = D == 1 ? SincKernel : KaiserKernel
  _ks = KrylovSolver(ks.preconditioner, _k, 0.0, ks.maxit)
  gen_kernel(_ks, pts, Ω)
end


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

function fouriertransform(sk::SincKernel{D}, wv::Vector{SVector{D,Float64}}) where{D}
  fouriertransform.(Ref(sk), wv)
end

function fouriertransform(sk::SincKernel{1}, w::Vector{Float64}) 
  [fouriertransform(sk, SA[wj]) for wj in w]
end

function gen_kernel(ks::KrylovSolver{P,SincKernel}, 
                    pts::Vector{SVector{D,Float64}}, Ω) where{P,D}
  SincKernel(Ω, ks.perturbation)
end

function gen_kernel(ks::SketchSolver{SincKernel}, 
                    pts::Vector{SVector{D,Float64}}, Ω) where{D}
  SincKernel(Ω, 0.0)
end



#
# GaussKernel: works well in _any_ dimension because it is not oscillatory. But
# because the FT decays so rapidly, you need to pick the bandwidth high enough
# that you don't effectively zero out high-frequencies that the window does need
# to resolve.
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
GaussKernel(M::Float64; perturbation=1e-8) = GaussKernel((M,); perturbation=perturbation)

function (gk::GaussKernel{D})(x::SVector{D,Float64}, y::SVector{D,Float64}) where{D}
  inner = sum(j->abs2(gk.Mv[j]*(x[j] - y[j])), 1:D)
  out   = exp(-pi*inner)
  x == y ? out + gk.perturbation : out
end

function fouriertransform(gk::GaussKernel{D}, w::SVector{D,Float64}) where{D}
  inner = sum(j->abs2(w[j]/gk.Mv[j]), 1:D)
  exp(-pi*inner)/gk.prodmv
end

function fouriertransform(gk::GaussKernel{D}, wv::Vector{SVector{D,Float64}}) where{D}
  fouriertransform.(Ref(gk), wv)
end

function fouriertransform(gk::GaussKernel{1}, wv::Vector{Float64})
  [fouriertransform(gk, SA[w]) for w in wv]
end

function gen_kernel(ks::KrylovSolver{P,GaussKernel},
                    pts::Vector{SVector{D,Float64}}, Ω) where{P,D}
  GaussKernel(Ω; perturbation=ks.perturbation)
end

function gen_kernel(ks::SketchSolver{GaussKernel},
                    pts::Vector{SVector{D,Float64}}, Ω) where{D}
  GaussKernel(Ω.*0.6; perturbation=0.0)
end

#
# MaternKernel: the Matern covariance function, supported in any dimension. The
# marginal variance is fixed at 1, but the range and smoothness parameters can
# be selected. At the moment, I don't think this is better in any circumstance
# than SincKernel or GaussKernel. But putting it here just in case. Probably
# should restrict nu to hit the fast paths for besselk.
#
struct MaternKernel
  rho::Float64
  nu::Float64
  perturbation::Float64
end

function (mk::MaternKernel)(x, y) 
  matern_cov(x-y, (1.0, mk.rho, mk.nu)) + Float64(x==y)*mk.perturbation
end

fouriertransform(mk::MaternKernel, w::Float64) = matern_sdf(w, (1.0, mk.rho, mk.nu))
fouriertransform(mk::MaternKernel, w::SVector{D,Float64}) where{D} = matern_sdf(w, (1.0, mk.rho, mk.nu))

function fouriertransform(mk::MaternKernel, wv::AbstractVector)
  [fouriertransform(mk, w) for w in wv]
end

function gen_kernel(ks::KrylovSolver{P,MaternKernel},
                    pts::Vector{SVector{1,Float64}}, Ω) where{P}
  MaternKernel(0.75*inv(maximum(Ω)), 4.5, ks.perturbation)
end

function gen_kernel(ks::KrylovSolver{P,MaternKernel},
                    pts::Vector{SVector{2,Float64}}, Ω) where{P}
  MaternKernel(0.5*sqrt(inv(maximum(Ω))), 4.5, ks.perturbation)
end

# Only implemented in 1D for now.
function kernel_tol_radius(::Val{1}, mk::MaternKernel, tol::Float64)
  b  = mk.rho
  m0 = mk(0.0, 0.0)
  while mk(0.0, b) > tol
    b *= 2
  end
  gss(t->(mk(0.0, t)/m0 - tol)^2, 0.0, b)
end

#
# KaiserKernel: the Kaiser function in kernel form.
#
struct KaiserKernel{D}
  kv::NTuple{D,Kaiser}
  perturbation::Float64
end

function (kk::KaiserKernel{D})(x::SVector{D,Float64}, y::SVector{D,Float64}) where{D}
  prod(j->kk.kv[j](x[j]-y[j]), 1:D) + Float64(x==y)*kk.perturbation
end

function fouriertransform(kk::KaiserKernel{D}, w::SVector{D,Float64}) where{D}
  prod(j->fouriertransform(kk.kv[j], w[j]), 1:D)
end

function fouriertransform(kk::KaiserKernel{D}, wv::Vector{SVector{D,Float64}}) where{D}
  fouriertransform.(Ref(kk), wv)
end

function gen_kernel(ks::KrylovSolver{P,KaiserKernel},
                    pts::Vector{SVector{D,Float64}}, Ω) where{P,D}
  D == 1 && @warn "The Kaiser preconditioner kernel in 1D is not advisable---it can easily result in not controlling frequencies small than Ω. Please use a SincKernel instead."
  kv = ntuple(D) do j
    ptsj     = getindex.(pts, j)
    (_a, _b) = extrema(x->x[1], ptsj)
    ab       = (_a + _b)/2
    (a, b)   = (_a - ab, _b-ab)
    Kaiser(Ω[j]*5, a=a, b=b)
  end
  KaiserKernel(kv, ks.perturbation)
end

