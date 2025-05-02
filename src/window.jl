
# This class of window function implements scalar evaluation of the window and
# its Fourier transform.
abstract type ClosedFormWindow end

# This one doesn't: the only thing that an ImplicitWindow offers is a valid RHS.
abstract type ImplicitWindow end

# REQUIRED interface for any window function object.
#
# linsys_rhs(win, frequency_grid) -> Vector{ComplexF64}
# default_Ω(pts, win) -> Float64  __OR__ window_support(win) -> Tuple{Float64, Float64}.
#
# ADDITIONALLY, for a ClosedFormWindow:
#
# bandwidth(win) -> Float64
# (win::YourWindow)(x) -> Float64                                      (scalar evaluation)
# IrregularSpectra.fouriertransform(win::YourWindow, ω) -> ComplexF64  (scalar FT)
#
# OPTIONALLY, if you want to use the standard default_Ω function:
#
# window_support(win) -> Tuple{Float64, Float64}


#
# Kaiser window:
#

struct Kaiser <: ClosedFormWindow
  beta::Float64
  normalizer::Float64
  a::Float64
  b::Float64
end

window_support(ka::Kaiser) = (ka.a, ka.b)

"""
Kaiser(W; a, b)

A Kaiser window function with (half-)bandwidth W and support on [a, b].
"""
function Kaiser(W; a=0.0, b=1.0) 
  beta   = W*pi*abs(b-a)
  (s, c) = (1/(b-a), -a/(b-a)-1/2)
  nr     = sqrt(quadgk(x->unitkaiser(s*x + c, beta)^2, a, b, atol=1e-10)[1])
  Kaiser(beta, nr, a, b)
end

bandwidth(ka::Kaiser) = (ka.beta/(pi*abs(ka.b-ka.a)))

# on [-1/2, 1/2]!
function unitkaiser(x, beta)
  half = one(x)/2
  -half <= x <= half || return zero(x)
  besseli(0, beta*sqrt(1 - (2*x)^2))
end

function (ka::Kaiser)(x)
  (a, b) = (ka.a, ka.b)
  (s, c) = (1/(b-a), -a/(b-a)-1/2)
  unitkaiser(s*x + c, ka.beta)/ka.normalizer
end

function unitkbwindow_ft(w, beta)
  pilf2   = (pi*w)^2
  (b, b2) =  (beta, beta^2)
  if pilf2 < b2
    arg = sqrt(b2 - pilf2)
    return sinh(arg)/arg
  else
    arg = sqrt(pilf2 - b2)
    return sinc(arg/pi)
  end
end

function fouriertransform(ka::Kaiser, w::Float64)
  (a, b) = (ka.a, ka.b)
  (s, c) = (1/(b-a), -a/(b-a)-1/2)
  (cispi(2*w*c/s)/s)*unitkbwindow_ft(w/s, ka.beta)/ka.normalizer
end

linsys_rhs(ka::Kaiser, frequency_grid) = fouriertransform.(Ref(ka), frequency_grid)


#
# Prolate function (one dimension):
#

"""
Prolate1D(bandwidth::Float64, intervals::Vector{NTuple{2,Float64}})

A (generalized) prolate function with support on `intervals` and half-bandwidth
`bandwidth`. Unlike a closed-form window like the Kaiser, this function and its
Fourier transform are only available by numerical solving an integral eigenvalue
problem. Because of this, scalar evaluation methods like `(p::Prolate1D)(x)` or
`IrregularSpectra.fouriertransform(p::Prolate1D, ω::Float64)` are intentionally
not implemented.
"""
struct Prolate1D <: ImplicitWindow
  bandwidth::Float64
  intervals::Vector{NTuple{2,Float64}}
end

bandwidth(p::Prolate1D) = p.bandwidth

slepkernel(xmy::Float64, bw::Float64) = sinc(2*bw*xmy)

function slepkernel(xmy::SVector{2,Float64}, bw::Float64)
  nx = 2*pi*bw*norm(xmy)
  iszero(nx) && return 0.5
  Bessels.besselj1(nx)/nx
end

function krylov_nquad(pts::Vector{Float64}, p::Prolate1D)
  out = maximum(p.intervals) do (aj, bj)
    count(x-> aj <= x <= bj, pts)
  end
  4*out + 100
end

function linsys_rhs(p::Prolate1D, wgrid::AbstractVector{Float64})
  Ω = maximum(abs, wgrid)
  (nodes, weights) = segment_glquadrule_nyquist(p.intervals, Ω)
  # Step 1: get the prolate in the time domain.
  (cnodes, cweights, cslep) = prolate_timedomain(p, max(512, 5*prolate_minimal_m(p)))
  slep = prolate_interpolate(p, cnodes, cweights, cslep, nodes, weights)
  # Step 2: compute their CFT.
  nufftop = NUFFT3(nodes, collect(wgrid.*(2*pi)), false, 1e-15)
  spectra = Vector{ComplexF64}(undef, length(wgrid))
  mul!(spectra, nufftop, complex(weights.*slep))
end

function prolate_interpolate(p::Prolate1D, coarse_nodes, coarse_weights, 
                             coarse_values, fine_nodes, fine_weights)
  # TODO (cg 2025/03/28 18:16): change FastSinc to not be hard-coded symmetric
  # to speed this up (although the point is that this matrix is very
  # rectangular, so not crucial).
  S = [slepkernel(x-y, p.bandwidth) for y in fine_nodes, x in coarse_nodes]
  s = S*(coarse_values.*coarse_weights)
  s ./= sqrt(dot(fine_weights, abs2.(s)))
end

function prolate_fromrule(w, nodes, weights)
  M    = [slepkernel(tj-tk, w) for tj in nodes, tk in nodes]
  Dw   = Diagonal(sqrt.(weights))
  A    = Symmetric(Dw*M*Dw)
  Ae   = eigen!(A)
  slep = real(Ae.vectors[:,end])
  ldiv!(Dw, slep)
  slep ./= sqrt(dot(weights, abs2.(slep)))
  slep .*= sign(slep[findmax(abs, slep)[2]])
  slep
end

function prolate_timedomain(p::Prolate1D, m; maxsize=5000)
  (nodes, weights) = segment_glquadrule(p.intervals, m)
  length(nodes) > maxsize && throw(error("Size limit reached for exact prolate computation. Perhaps your refinement has failed?"))
  (nodes, weights, prolate_fromrule(p.bandwidth, nodes, weights))
end

function prolate_minimal_m(p::Prolate1D)
  minimum(p.intervals) do (aj, bj)
    2*p.bandwidth*(bj - aj)*4
  end
end

function default_Ω(pts::Vector{Float64}, g::Prolate1D; check=false)
  minimum(g.intervals) do (aj, bj)
    nj = count(x-> aj <= x <= bj, pts)
    0.8*nj/(4*(bj - aj))
  end
end

#
# Sine taper (1D):
#

struct Sine <: ClosedFormWindow
  a::Float64
  b::Float64
end

function default_Ω(pts::Vector{Float64}, sw::Sine)
  @info "Since the sine window is not very concentrated, the default Ω is slightly lower than with windows like the Kaiser. But you can often resolve higher Ω than this default without huge blowup, so feel free to experiment with manually setting Ω yourself." maxlog=1
  0.7*length(pts)/(4*(sw.b-sw.a))
end

window_support(sw::Sine) = (sw.a, sw.b)

#=
function bandwidth(sw::Sine) 
  error("The sine window FT decays like 1/ω^2, so its bandwidth is too large for applications that assume approximate compact support for the window FT.")
end
=#

# on [-1/2, 1/2]
function unit_sinewindow(x)
  (-1/2 < x < 1/2) || return zero(x)
  sqrt(2)*sinpi(x+1/2)
end

function (sw::Sine)(x)
  (a, b) = (sw.a, sw.b)
  (s, c) = (1/(b-a), -a/(b-a) - 1/2)
  unit_sinewindow(s*x + c)*sqrt(s)
end

unit_sinewindow_ft(ω) = sqrt(2)*2*cospi(ω)/(pi*(1 - 4*ω^2))

function fouriertransform(sw::Sine, ω)
  (a, b) = (sw.a, sw.b)
  (s, c) = (1/(b-a), -a/(b-a) - 1/2)
  (cispi(2*ω*c/s)/s)*unit_sinewindow_ft(ω/s)*sqrt(s)
end

linsys_rhs(sw::Sine, fgrid) = fouriertransform.(Ref(sw), fgrid)

#
# Tensor product of two 1D windows:
#

struct TensorProduct2DWindow{W1,W2}
  s1::W1
  s2::W2
end

function linsys_rhs(sp::TensorProduct2DWindow, fgrid)
  unique_f1 = sort(unique(getindex.(fgrid, 1)))
  unique_f2 = sort(unique(getindex.(fgrid, 2)))
  w1_vals   = Dict(zip(unique_f1, linsys_rhs(sp.s1, unique_f1)))
  w2_vals   = Dict(zip(unique_f2, linsys_rhs(sp.s2, unique_f2)))
  [w1_vals[fj[1]]*w2_vals[fj[2]] for fj in fgrid]
end

# TODO (cg 2025/04/26 13:44): figure out a better default Ω here that is safe
# enough to give weights with a decent norm but also not needlessly cautious.
function default_Ω(pts::Vector{SVector{2,Float64}}, sp::TensorProduct2DWindow)
  mΩ = min(default_Ω(getindex.(pts, 1), sp.s1), default_Ω(getindex.(pts, 2), sp.s2))
  Ω  = sqrt(mΩ)/2
  (Ω, Ω)
end

