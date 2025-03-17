
window_support(winfun) = (winfun.a, winfun.b)

struct FourierTransform{F}
  fn::F
end

struct Kaiser
  beta::Float64
  normalizer::Float64
  a::Float64
  b::Float64
end

"""
Kaiser(beta; a, b)

A Kaiser window function with shape parameter beta and support on [a, b].
"""
function Kaiser(beta; a=0.0, b=1.0) 
  (s, c) = (1/(b-a), -a/(b-a)-1/2)
  nr     = sqrt(quadgk(x->unitkaiser(s*x + c, beta)^2, a, b, atol=1e-10)[1])
  Kaiser(beta, nr, a, b)
end

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

function (fka::FourierTransform{Kaiser})(w)
  (a, b) = (fka.fn.a, fka.fn.b)
  (s, c) = (1/(b-a), -a/(b-a)-1/2)
  (cispi(-2*w*c/s)/s)*unitkbwindow_ft(w/s, fka.fn.beta)/fka.fn.normalizer
end

