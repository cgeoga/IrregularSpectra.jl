
# This class of window function implements scalar evaluation of the window and
# its Fourier transform.
abstract type ClosedFormWindow end

# This one doesn't: the only thing that an ImplicitWindow offers is a valid RHS.
abstract type ImplicitWindow end

# REQUIRED interface for any window function object.
#
# window_support(win) -> Tuple{Float64, Float64}
# bandwidth(win) -> Float64
# linsys_rhs(win, frequency_grid) -> Vector{ComplexF64}
#
# ADDITIONALLY, for a ClosedFormWindow:
#
# (win::YourWindow)(x) -> Float64                                      (scalar evaluation)
# IrregularSpectra.fouriertransform(win::YourWindow, ω) -> ComplexF64  (scalar FT)


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

window_support(p::Prolate1D) = (minimum(minimum, p.intervals),
                                maximum(maximum, p.intervals))

bandwidth(p::Prolate1D) = p.bandwidth

function linsys_rhs(p::Prolate1D, wgrid::AbstractVector{Float64}; tol=1e-10)
  # Step 0: get a quadrature rule.
  Ω      = maximum(abs, wgrid)
  (nodes, weights, nwv) = segment_glquadrule_nyquist(p.intervals, Ω)
  # Step 1: get the prolate functions in the time domain. For reproducibility
  # and testing reasons, we also provide a manual initialization.
  M    = [sinc(2*p.bandwidth*(tj-tk)) for tj in nodes, tk in nodes]
  Dw   = Diagonal(sqrt.(weights))
  A    = Symmetric(Dw*M*Dw)
  init = begin # very heuristic!
    lengths = map(iv -> abs(iv[2] - iv[1]), p.intervals)
    normalize!(lengths, 2)
    kaisers = map(enumerate(p.intervals)) do (j, (aj, bj))
      ka = Kaiser(bandwidth(p)*pi*(bj - aj), a=aj, b=bj)
      ka.(nwv[j][1]).*lengths[j]
    end
    Dw*reduce(vcat, kaisers)
  end
  pe = partialeigen(partialschur(A, nev=1, v1=init, tol=tol)[1])
  Ae = (values=pe[1], vectors=pe[2]) 
  isempty(Ae.vectors) && throw(error("No convergence for any eigenvectors in discretized concentration problem!"))
  ldiv!(Dw, Ae.vectors)
  # sanitize the output: sometimes if you resolve more eigenpairs than
  # requested, the output won't have the shape you think. So this routine simply
  # sorts things how you would expect and standardizes the output.
  eix  = findmax(abs, Ae.values)[2]
  slep = real(Ae.vectors[:,eix])
  # quick normalization to make the prolates have unit L2 norm:
  slep ./= sqrt(dot(weights, abs2.(slep)))
  # Step 2: compute their CFT.
  nufftop  = nudftmatrix(wgrid, nodes, -1)
  spectra  = Vector{ComplexF64}(undef, length(wgrid))
  mul!(spectra, nufftop, complex(weights.*slep))
end

