
function default_Ω(pts, g; check=true)
  # Note that window_support(g::YourWindow) may not be implemented, in which
  # case you either need to implement it or write a custom default_Ω.
  (a, b) = window_support(g)
  if check
    (_a, _b) = extrema(pts)
    _a < a && @warn "Your window function g(x) has support [$a, $b] and you have a point $(_a) < $a."
    _b > b && @warn "Your window function g(x) has support [$a, $b] and you have a point $(_b) > $b."
  end
  0.8*length(pts)/(4*(b-a))
end

"""
`(Ω, weights) = window_quadrature_weights(pts::Vector{Float64}, g; kwargs...)`

Computes the weights {α_j}_{j=1}^n such that

H_{α}(ω) ≈ G(ω)

for |ω| ≤ Ω. The default Ω is chosen to be 90% of the continuous Nyquist
frequency, which is n/(4*(b-a)). In the case where Ω = O(n), with the current
routine this computation will scale like O(n³). If Ω is fixed and doesn't grow
with n, then it will scale like O(n log n).

The object `g::G` can be any structure represnting a window function. It needs
to implement the methods described at the top of `./src/window.jl`.

Keyword arguments are as follows:

- `method = :krylov`: the numerical method used to obtain the weights. See the
paper for full details, but this is a potentially very large least squares problem
with a rank-deficient design matrix that needs to be solved. Options are:

  - `:krylov`: the default choice, which uses `Krylov.lsmr` to compute the weights
    in an entirely matrix-free way. We highly suggest `verbose=true` here, because
    if you have asked for an impossible window function the convergence will be slow.
    In our experience, if it takes more than ~100 iterations, something isn't right.

  - `:sketch`: uses `LowRankApprox` to obtain a partial pivoted QR factorizaton for
    the associated nonuniform Fourier matrix. If `Ω` is small compared to `n`, this
    may be the fastest option. It is necessary to obtain this partial factorization
    to very high accuracy, though, so the worst case here is when the Fourier matrix
    is unitary. In that case, you'll basically do O(log(n)) many full QR factorizations
    and it will be very slow.

  - `:dense`: uses a simple `qr(F, ColumnNorm())` on the nonuniform Fourier matrix.
    This will almost never be the fastest option, but we offer it for those who
    are exploring or debugging.

- `Ω = default_Ω(pts, g)`: the highest frequency that the weights will attempt
to resolve. This defaults to 80% of the Nyquist frequency, but can be adaptively
reduced if the norm of the weights is too high.

- `max_wt_norm = Inf`: the maximum permissible norm of the weight vector. This
norm has a big impact on the size of the aliasing bias, so it can sometimes be
advantageous to reduce `Ω` in exchange for a smaller norm. If this is set to a
finite value and the computed weights exceed it, then `Ω` is reduced as `Ω *=
reduction_factor` and the weights are recomputed.

- `reduction_factor = 0.9`: this quantity determines how aggressively we shrink
`Ω` in the case when the computed weights exceed `max_wt_norm`. Making it larger
may save you on the cost of recomputing weights, but it may mean your ultimate
`Ω` is lower than it could have been.

- `verbose = true`: whether or not to print output.

**See also:** the function `matern_frequency_selector` takes points and a window
function and returns both weights and the highest ``safe" frequency to estimate
based on a user-provided tolerance for the relative size in aliasing bias.
"""
function window_quadrature_weights(pts::Vector{Float64}, g; solver=default_solver(pts),
                                   Ω=default_Ω(pts, g), max_wt_norm=Inf, 
                                   min_Ω=0.05*Ω, reduction_factor=0.9, verbose=true)
  wts    = solve_linsys(pts, g, Ω, solver, verbose=verbose)
  verbose && @printf "||α||₂:             %1.5e\n" norm(wts)
  while norm(wts) > max_wt_norm
    Ω     *= reduction_factor
    Ω < min_Ω && throw(error("Could not achieve ||α||₂ < $max_wt_norm for Ω > $min_Ω."))
    wts    = solve_linsys(pts, g, Ω, solver, verbose=verbose)
    verbose && @printf "||α||₂:             %1.5e\n" norm(wts)
  end
  (Ω, wts)
end

"""
estimate_sdf(pts::Vector{Float64}, data, g, frequencies; Ω=default_Ω(pts, g),
             wts=nothing, kwargs...)

Estimates the spectral density for the stationary process sampled at locations
`pts` and with values given by `data` at frequencies `frequencies`. Each column
in `data` is treated as an iid sample of the process and averaged in the final
estimator. `Ω` is the resolution maximum for the window quadrature weights. 

As of now, this function does not check the frequencies you are estimating compared
to the highest resolvable frequency Ω! A safer interface is under development.

Any provided keyword arguments are passed to `window_quadrature_weights` if `wts` is
`nothing`, and ignored if `wts` is provided. See the docstrings for `window_quadrature_weights`
for details on available kwargs. If `wts` is provided, the `Ω` arg also does nothing.
"""
function estimate_sdf(pts::Vector{Float64}, data, g, frequencies; 
                      Ω=default_Ω(pts, g), wts=nothing, kwargs...)
  if isnothing(wts)
    (Ω, wts) = window_quadrature_weights(pts, g; Ω=Ω, kwargs...)
  end
  fs  = NUFFT3(pts, collect(frequencies.*(2*pi)), true, 1e-15)
  out = zeros(ComplexF64, length(frequencies), size(data, 2))
  mul!(out, fs, complex(Diagonal(wts)*data))
  mean(x->abs2.(x), eachcol(out))
end

