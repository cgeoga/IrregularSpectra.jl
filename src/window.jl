
# TODO (cg 2025/03/21 13:56): Maybe there should be some sort of type hierarchy
# like
#
# abstract type ClosedFormWindow end
# abstract type ImplicitWindow end
# 
# to signify the difference between a window that implements
# fouriertransform(window, freq(s)) and one that only offers linsys_rhs.


# REQUIRED interface for a window function object.
#
# window_support(win) -> Tuple{Float64, Float64}
# bandwidth(win) -> Float64
# linsys_rhs(win, frequency_grid) -> Vector{ComplexF64}


#
# Kaiser window:
#

struct Kaiser
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

struct Prolate1D
  intervals::Vector{NTuple{2,Float64}}
  bandwidth::Float64
end

window_support(p::Prolate1D) = (minimum(minimum, p.intervals),
                                maximum(maximum, p.intervals))

bandwidth(p::Prolate1D) = p.bandwidth

function linsys_rhs(p::Prolate1D, wgrid::AbstractVector{Float64}; tol=1e-10)
  # Step 0: get a quadrature rule.
  Ω      = maximum(abs, wgrid)
  (nodes, weights, nwv) = segment_glquadrule_nyquist(p.intervals, Ω)
  # Step 1: get the prolate functions in the time domain.
  M       = [sinc(2*p.bandwidth*(tj-tk)) for tj in nodes, tk in nodes]
  Dw      = Diagonal(sqrt.(weights))
  A       = Symmetric(Dw*M*Dw) + 1e-10I
  pe      = partialeigen(partialschur(A, nev=1, tol=tol)[1])
  Ae     = (values=pe[1], vectors=pe[2]) 
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

