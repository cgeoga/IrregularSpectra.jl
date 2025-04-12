
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
# ADDITIONALLY, if you want to use the Krylov weight computation:
#
# krylov_nquad(pts, win) -> Int64
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
Kaiser(beta; a, b)

A Kaiser window function with shape parameter beta and support on [a, b].
"""
function Kaiser(beta; a=0.0, b=1.0) 
  (s, c) = (1/(b-a), -a/(b-a)-1/2)
  nr     = sqrt(quadgk(x->unitkaiser(s*x + c, beta)^2, a, b, atol=1e-10)[1])
  Kaiser(beta, nr, a, b)
end

bandwidth(ka::Kaiser) = (ka.beta/(pi*abs(ka.b-ka.a)))

krylov_nquad(pts, ka::Kaiser) = 4*length(pts) + 100

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

linsys_rhs(ka::Kaiser, frequency_grid) = fouriertransform(ka, frequency_grid)


#
# Prolate function (one dimension):
#

"""
Prolate1D(intervals::Vector{NTuple{2,Floa564}}, bandwidth::Float64)

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

function krylov_nquad(pts, p::Prolate1D)
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
  S = [sinc(2*p.bandwidth*(x-y)) for y in fine_nodes, x in coarse_nodes]
  s = S*(coarse_values.*coarse_weights)
  s ./= sqrt(dot(fine_weights, abs2.(s)))
end

function prolate_timedomain(p::Prolate1D, m; maxsize=5000)
  (nodes, weights) = segment_glquadrule(p.intervals, m)
  length(nodes) > maxsize && throw(error("Size limit reached for exact prolate computation. Perhaps your refinement has failed?"))
  sqrtwts = sqrt.(weights)
  Dw = Diagonal(sqrtwts)
  A  = Symmetric([sinc(2*p.bandwidth*(nodes[j] - nodes[k]))*sqrtwts[j]*sqrtwts[k]
                  for j in eachindex(nodes), k in eachindex(nodes)])
  Ae   = eigen!(A)
  slep = real(Ae.vectors[:,end])
  ldiv!(Dw, slep)
  slep ./= sqrt(dot(weights, abs2.(slep)))
  slep .*= sign(slep[findmax(abs, slep)[2]])
  (nodes, weights, slep)
end

function prolate_minimal_m(p::Prolate1D)
  minimum(p.intervals) do (aj, bj)
    2*p.bandwidth*(bj - aj)*4
  end
end

# An automatically refining alternative.
#
# TODO (cg 2025/03/28 17:47): for larger bandwidths, I think this is just not
# stable because the eigenvalue problem itself seems very numerically
# challenging. I think we are getting a linear combination of effectively
# equally concentrated vectors, but it is throwing off these pointwise
# assessments of convergence.
function prolate_timedomain(p::Prolate1D; m_init=prolate_minimal_m(p), tol=0.0025)
  (nodes_m,  weights_m,  slep_m)  = prolate_timedomain(p, m)
  (nodes_2m, weights_2m, slep_2m) = prolate_timedomain(p, 2*m)
  slep_itp = prolate_interpolate(p, nodes_m, weights_m, slep_m, nodes_2m, weights_2m)
  while maximum(abs, slep_itp - slep_2m) > tol
    m *= 2
    nodes_m   = nodes_2m
    weights_m = weights_2m
    slep_m    = slep_2m
    (nodes_2m, weights_2m, slep_2m) = prolate_timedomain(p, 2*m)
    slep_itp  = prolate_interpolate(p, nodes_m, weights_m, slep_m, 
                                    nodes_2m, weights_2m)
  end
  (nodes_2m, weights_2m, slep_2m) 
end

function default_Ω(pts, g::Prolate1D; check=false)
  minimum(g.intervals) do (aj, bj)
    nj = count(x-> aj <= x <= bj, pts)
    0.8*nj/(4*(bj - aj))
  end
end

