
function default_Ω(pts, g; check=true)
  (a, b)   = window_support(g)
  if check
    (_a, _b) = extrema(pts)
    _a < a && @warn "Your window function g(x) has support [$a, $b] and you have a point $(_a) < $a."
    _b > b && @warn "Your window function g(x) has support [$a, $b] and you have a point $(_b) > $b."
  end
  0.8*length(pts)/(4*(b-a))
end

"""
`window_quadrature_weights(pts::Vector{Float64}, g; Ω=default_Ω(pts, g), verbose=true)`

Computes the weights {α_j}_{j=1}^n such that

H_{α}(ω) ≈ G(ω)

for |ω| ≤ Ω. The default Ω is chosen to be 90% of the continuous Nyquist
frequency, which is n/(4*(b-a)). In the case where Ω = O(n), with the current
routine this computation will scale like O(n³). If Ω is fixed and doesn't grow
with n, then it will scale like O(n log n).

The object `g::G` can be any structure represnting a window function. It needs
to implement the methods described at the top of `./src/window.jl`.

**See also:** the function `matern_frequency_selector` takes points and a window
function and returns both weights and the highest ``safe" frequency to estimate
based on a user-provided tolerance for the relative size in aliasing bias.
"""
function window_quadrature_weights(pts::Vector{Float64}, g; method=:sketch,
                                   Ω=default_Ω(pts, g), verbose=true)
  wgrid  = range(-Ω, Ω, length=2*length(pts))
  b      = linsys_rhs(g, wgrid) 
  wts    = solve_linsys(pts, wgrid, b, method=method)
  if verbose
    @printf "Weight diagnostics:\n"
    @printf "\t - Rank of reduced QR: %i\n" size(fsoqr.Q)[2]
    @printf "\t - ||α||₂:             %1.5e\n" norm(wts)
  end
  wts
end

"""
estimate_sdf(pts::Vector{Float64}, data, g, frequencies; Ω=default_Ω(pts, g))

Estimates the spectral density for the stationary process sampled at locations
`pts` and with values given by `data` at frequencies `frequencies`. Each column
in `data` is treated as an iid sample of the process and averaged in the final
estimator. `Ω` is the resolution maximum for the window quadrature weights. 

As of now, this function does not check the frequencies you are estimating compared
to the highest resolvable frequency Ω! A safer interface is under development.
"""
function estimate_sdf(pts::Vector{Float64}, data, g, frequencies; 
                      Ω=default_Ω(pts, g), wts=window_quadrature_weights(pts, g; Ω=Ω))
  fs  = NUFFT3(pts, frequencies.*(2*pi), true, 1e-15)
  out = zeros(ComplexF64, length(frequencies), size(data, 2))
  mul!(out, fs, complex(Diagonal(wts)*data))
  mean(x->abs2.(x), eachcol(out))
end

