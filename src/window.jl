
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
  # TODO (cg 2025/05/10 11:08): this is probably possible to get in closed form.
  nr     = sqrt(quadgk(x->unitkaiser(s*x + c, beta)^2, a, b, rtol=1e-10)[1])
  Kaiser(beta, nr, a, b)
end

bandwidth(ka::Kaiser) = (ka.beta/(pi*abs(ka.b-ka.a)))

# on [-1/2, 1/2]!
function unitkaiser(x, beta)
  half = one(x)/2
  -half <= x <= half || return zero(x)
  innerx = sqrt(1-(2*x)^2)
  besselix(0, beta*innerx)*exp(-beta*(1-innerx))
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
    arg < 10 && return exp(-beta)*sinh(arg)/arg
    # this is equivalent to exp(-beta)*sinh(arg)/arg, but the sinh arg is huge
    # and so we manually expand some things out to avoid losing digits.
    return exp(arg-beta)*(1 - exp(-2*arg))/(2*arg)
  else
    arg = sqrt(pilf2 - b2)
    return exp(-b)*sinc(arg/pi)
  end
end

function fouriertransform(ka::Kaiser, w::Float64)
  (a, b) = (ka.a, ka.b)
  (s, c) = (1/(b-a), -a/(b-a)-1/2)
  (cispi(2*w*c/s)/s)*unitkbwindow_ft(w/s, ka.beta)/ka.normalizer
end

fouriertransform(ka::Kaiser, w::SVector{1,Float64}) = fouriertransform(ka, w[1])

linsys_rhs(ka::Kaiser, frequency_grid) = hcat(fouriertransform.(Ref(ka), frequency_grid))


#
# Sine taper (1D):
#

struct Sine <: ClosedFormWindow
  a::Float64
  b::Float64
end

function default_Ω(pts::Vector{Float64}, sw::Sine)
  @info "Since the sine window is not very concentrated, the default Ω is slightly lower than with windows like the Kaiser. But you can often resolve higher Ω than this default without huge blowup, so feel free to experiment with manually setting Ω yourself." maxlog=1
  (is_gridded, gridded_Ω) = gappy_grid_Ω(pts)
  is_gridded && return gridded_Ω
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

linsys_rhs(sw::Sine, fgrid) = hcat(fouriertransform.(Ref(sw), fgrid))


#
# Prolate function (one dimension):
#

"""
`Prolate1D(bandwidth::Float64, intervals::Vector{NTuple{2,Float64}})`

A (generalized) prolate function with support on `intervals` and half-bandwidth
`bandwidth`. Unlike a closed-form window like the Kaiser, this function and its
Fourier transform are only available by numerical solving an integral eigenvalue
problem. Because of this, scalar evaluation methods like `(p::Prolate1D)(x)` or
`IrregularSpectra.fouriertransform(p::Prolate1D, ω::Float64)` are intentionally
not implemented.

`Prolate1D` has the construction method

`Prolate1D(intervals; bandwidth=default_bandwidth(intervals))`

which automatically selects the bandwidth. The number of tapers is selected
automatically based on being sufficiently concentrated.
"""
struct Prolate1D <: ImplicitWindow
  bandwidth::Float64
  intervals::Vector{NTuple{2,Float64}}
end

bandwidth(p::Prolate1D) = p.bandwidth

function default_prolate_bandwidth(intervals::Vector{NTuple{2,Float64}}) 
  minimum(intervals) do ivj
    (aj, bj) = ivj
    5.0/(bj - aj)
  end
end

function Prolate1D(intervals::Vector{NTuple{2,Float64}};
                   bandwidth=default_prolate_bandwidth(intervals))
  Prolate1D(bandwidth, intervals)
end


slepkernel(xmy::Float64, bw::Float64) = sinc(2*bw*xmy)

function linsys_rhs(p::Prolate1D, wgrid::AbstractVector{Float64})
  # Step 1: compute the prolate on a coarse grid that just resolves the Nyquist
  # frequency using a dense eigendecomposition.
  (cnodes, cweights) = segment_glquadrule_nyquist(p.intervals, 2*p.bandwidth) 
  cslep = prolate_fromrule(p.bandwidth, cnodes, cweights)
  # Step 2: interpolate that coarse prolate up to a sufficiently large fine grid
  # that you can resolve all the oscillations in wgrid.
  (nodes, weights) = segment_glquadrule_nyquist(p.intervals, maximum(abs, wgrid))
  slep  = prolate_interpolate(p.bandwidth, cnodes, cweights, cslep, nodes, weights)
  # Step 2: compute its CFT.
  nufftop = NUFFT3(collect(wgrid.*(2*pi)), nodes, -1)
  spectra = Matrix{ComplexF64}(undef, length(wgrid), size(slep, 2))
  mul!(spectra, nufftop, complex(weights.*slep))
end

# This is now just a fallback to use for debugging.
function slepian_operator(x1v, x2v, bandwidth)
  [slepkernel(x-y, bandwidth) for y in x1v, x in x2v]
end

function fast_slepian_operator(x1v::Vector{Float64}, x2v::Vector{Float64}, 
                               bandwidth)
  FastBandlimited(x1v, x2v, ω->inv(2*bandwidth), bandwidth)
end

function fast_slepian_operator(x1v::Vector{SVector{2,Float64}}, 
                               x2v::Vector{SVector{2,Float64}}, 
                               bandwidth)
  FastBandlimited(x1v, x2v, ω->inv(pi*bandwidth^2)/2, bandwidth; polar=true)
end

function prolate_interpolate(bandwidth, coarse_nodes, coarse_weights, 
                             coarse_values, fine_nodes, fine_weights)
  S = fast_slepian_operator(fine_nodes, coarse_nodes, bandwidth)
  s = S*(coarse_values.*coarse_weights)
  for sj in eachcol(s)
    sj ./= sqrt(dot(fine_weights, abs2.(sj)))
  end
  s
end

function _dense_prolate_fromrule(w, nodes, weights; concentration_tol=1e-8)
  M    = slepian_operator(nodes, nodes, w) 
  Dw   = Diagonal(sqrt.(weights))
  A    = Symmetric(Dw*M*Dw)
  Ae   = eigen!(A)
  rel_concs    = Ae.values
  rel_concs  ./= Ae.values[end]
  good_conc_ix = findfirst(>=(1-concentration_tol), rel_concs)
  prolates = Ae.vectors[:,good_conc_ix:end]
  ldiv!(Dw, prolates)
  for sj in eachcol(prolates)
    sj ./= sqrt(dot(weights, abs2.(sj)))
  end
  prolates
end

# this method is defined in the ArnoldiMethod.jl ext.
function _fast_prolate_fromrule end

function prolate_fromrule(w, nodes, weights; concentration_tol=1e-8)
  out = if length(nodes) > 3_000 && length(methods(_fast_prolate_fromrule)) > 0
    @info "Using `ArnoldiMethod.jl` to accelerate prolate computation..." maxlog=1
    _fast_prolate_fromrule(w, nodes, weights; concentration_tol=concentration_tol)
  else
    if length(nodes) > 3_000
      @warn "You will be computing a dense eigendecomposition of size $(length(nodes)). If this is too large or runs too slowly, consider ]adding and `using` `ArnoldiMethod.jl`, which will load an extension for computing these prolates in quasilinear time." maxlog=1
    end
    _dense_prolate_fromrule(w, nodes, weights; concentration_tol=concentration_tol)
  end
  @info "Obtained $(size(out, 2)) well-concentrated prolates for this domain and bandwidth $(w)."
  out
end

function prolate_minimal_m(p::Prolate1D)
  minimum(p.intervals) do (aj, bj)
    2*p.bandwidth*(bj - aj)*4
  end
end

function default_Ω(pts::Vector{Float64}, g::Prolate1D; check=false)
  (is_gridded, gridded_Ω) = gappy_grid_Ω(pts)
  is_gridded && return gridded_Ω
  minimum(g.intervals) do (aj, bj)
    nj = count(x-> aj <= x <= bj, pts)
    0.8*nj/(4*(bj - aj))
  end
end

#
# Prolate2D: for now, just on one solid rectangle. But that will be lifted in
# time.
#
struct Prolate2D <: ImplicitWindow
  bandwidth::Float64   # this is a _radius_.
  a::NTuple{2,Float64}
  b::NTuple{2,Float64}
end

bandwidth(p2d::Prolate2D) = p2d.bandwidth

function slepkernel(xmy::SVector{2,Float64}, bw::Float64)
  nx = 2*pi*bw*norm(xmy)
  iszero(nx) && return 0.5
  Bessels.besselj1(nx)/nx
end

function prolate_minimal_m(p::Prolate2D)
  m1 = 2*abs(p.b[1] - p.a[1])*4*p.bandwidth
  m2 = 2*abs(p.b[2] - p.a[2])*4*p.bandwidth
  max(m1, m2)
end

# TODO (cg 2025/05/22 13:49): make this a multitaper like in 1D.
function linsys_rhs(p::Prolate2D, wgrid::AbstractVector{SVector{2,Float64}})
  # Step 1: compute the prolate function on a quadrature grid that resolves the
  # bandwidth.
  corder = Int(ceil(max(32, prolate_minimal_m(p))))
  (cnodes, cweights) = glquadrule((corder, corder), p.a, p.b)
  cslep = prolate_fromrule(p.bandwidth, cnodes, cweights)
  # Step 2: obtain the prolate on a finer grid that can resolve the actual
  # oscillations of wgrid.
  Ωl1  = 4*Int(ceil(maximum(x->norm(x,1), wgrid)))
  (nodes, weights) = glquadrule((Ωl1, Ωl1), p.a, p.b)
  slep = prolate_interpolate(p.bandwidth, cnodes, cweights, cslep, nodes, weights)
  # Step 3: compute their CFT.
  nufftop = NUFFT3(collect(wgrid.*(2*pi)), nodes, -1)
  spectra = Matrix{ComplexF64}(undef, length(wgrid), size(cslep, 2))
  mul!(spectra, nufftop, complex(weights.*slep))
end

#=
function default_Ω(pts::Vector{SVector{2,Float64}}, p::Prolate2D)
  Ω1 = default_Ω(getindex.(pts, 1), Kaiser(p.bandwidth, a=p.a[1], b=p.b[1]))
  Ω2 = default_Ω(getindex.(pts, 2), Kaiser(p.bandwidth, a=p.a[2], b=p.b[2]))
  Ω  = sqrt(min(Ω1, Ω2))/2
  (Ω, Ω)
end
=#

#
# QuadratureRuleProlate: a prolate computed directly from a quadrature rule,
# which is assumed to be of sufficient order to resolve the highest
# oscillations. This is a _risky_ internal object for now.
#
struct QuadratureRuleProlate{D}
  bandwidth::Float64 # this is a _radius_.
  nodes::Vector{SVector{D,Float64}}
  weights::Vector{Float64}
  Ω::NTuple{D,Float64}
end
bandwidth(qp::QuadratureRuleProlate) = p2d.bandwidth

default_Ω(pts::Vector{SVector{D,Float64}}, p::QuadratureRuleProlate{D}) where{D} = p.Ω

function linsys_rhs(p::QuadratureRuleProlate{2}, 
                    wgrid::AbstractVector{SVector{2,Float64}})
  slep    = prolate_fromrule(p.bandwidth, p.nodes, p.weights)
  nufftop = NUFFT3(collect(wgrid.*(2*pi)), p.nodes, -1)
  spectra = Matrix{ComplexF64}(undef, length(wgrid), size(slep, 2))
  hcat(mul!(spectra, nufftop, complex(p.weights.*slep)))
end



#=
#
# Tensor product of two 1D windows:
#

struct TensorProduct2DWindow{W1,W2}
  s1::W1
  s2::W2
  zero_offlobe::Bool
end

TensorProduct2DWindow(s1, s2) = TensorProduct2DWindow(s1, s2, false)

function linsys_rhs(sp::TensorProduct2DWindow, fgrid)
  unique_f1 = sort(unique(getindex.(fgrid, 1)))
  unique_f2 = sort(unique(getindex.(fgrid, 2)))
  w1_vals   = Dict(zip(unique_f1, linsys_rhs(sp.s1, unique_f1)[:,end]))
  w2_vals   = Dict(zip(unique_f2, linsys_rhs(sp.s2, unique_f2)[:,end]))
  out       = [w1_vals[fj[1]]*w2_vals[fj[2]] for fj in fgrid]
  if sp.zero_offlobe
    bw = (bandwidth(sp.s1), bandwidth(sp.s2))
    for j in eachindex(fgrid, out)
      fj = fgrid[j]
      any(k->abs(fj[k]) > bw[k], 1:2) && (out[j] = zero(ComplexF64))
    end
  end
  hcat(out)
end

# TODO (cg 2025/04/26 13:44): figure out a better default Ω here that is safe
# enough to give weights with a decent norm but also not needlessly cautious.
function default_Ω(pts::Vector{SVector{2,Float64}}, sp::TensorProduct2DWindow)
  (is_gridded, gridded_Ω) = gappy_grid_Ω(pts)
  is_gridded && return gridded_Ω
  mΩ = min(default_Ω(getindex.(pts, 1), sp.s1), default_Ω(getindex.(pts, 2), sp.s2))
  Ω  = sqrt(mΩ)/2
  (Ω, Ω)
end
=#
