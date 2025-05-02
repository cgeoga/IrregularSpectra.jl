
function default_Ω(pts::Vector{Float64}, g; check=true)
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

function default_frequencies(pts::Vector{Float64}, g, Ω::Float64)
  fmax = Ω/2
  len  = hasmethod(bandwidth, (typeof(g),)) ? 4*fmax/bandwidth(g) : length(pts)/4
  collect(range(-fmax, fmax, length=Int(ceil(len))))
end

function default_frequencies(pts::Vector{SVector{D,Float64}}, 
                             g, Ω::NTuple{D,Float64}) where{D}
  fmax   = Ω./2
  totlen = hasmethod(bandwidth, (typeof(g),)) ? 4*fmax/bandwidth(g) : length(pts)/4
  len    = Int(ceil(totlen^(1/D)))
  fg1dv  = [range(-fmax[j], fmax[j], length=len) for j in eachindex(fmax)]
  fgd    = vec(SVector{D,Float64}.(Iterators.product(fg1dv...)))
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

- `solver = default_solver(pts)`: the numerical method used to obtain the
weights. See the paper for full details, but this is a potentially very large
least squares problem with a rank-deficient design matrix that needs to be
solved. Options are:

  - `KrylovSolver(::KrylovPreconditioner, pre_kernel::Type{K}, perturbation::Float64)`: 
    the default which uses `Krylov.lsmr` to compute the weights in an entirely
    matrix-free way. The `::KrylovPreconditioner` object is crucial for this to be accurate. 
    For small data sizes (maybe n ~ 5k or lower), the default `CholeskyPreconditioner()` 
    is your best option. But for larger data sizes, considering `]add`-ing
    `HMatrices.jl` and using the `HMatrixPreconditioner(atol, lu_atol)`, which
    will scale _much_ better. The second argument, `pre_kernel`, is a _type_ that
    that is an internal detail in assembling the preconditioned linear system. 
    In general, we suggest using `SincKernel` in 1D, and `GaussKernel` in 2+D. Finally,
    `perturbation` is another interal detail in the preconditioned linear system.
    A higher value will make certain preconditioners more stable, but slow down convergence.
    We suggest a default choice of `1e-8`. See the example files for a demo of using
    a custom `KrylovSolver`.

  - `DenseSolver()`: uses a simple `qr(F, ColumnNorm())` on the nonuniform
    Fourier matrix.  This will almost never be the fastest option, but we offer it
    for those who are exploring or debugging.

  - `SketchSolver()`: this is an extension that requires `LowRankApprox.jl`. In
    the setting where `Ω` is small or fixed and you are going to crank `n` up,
    the Fourier matrix has a bounded rank and using the NUFFT and sketching
    methods one can obtain weights rapidly with a partial QR. This may or may
    not be faster than the `KrylovSolver` with a good preconditioner, and the 
    circumstance where you should reach for this one is probably rare.

- `Ω = default_Ω(pts, g)`: the highest frequency that the weights will attempt
  to resolve. This defaults to 80% of the Nyquist frequency for most windows, but
  can be adaptively reduced if the norm of the weights is too high. See options below.
  Note that in D dimensions for D > 1, `Ω` is/must be a D-tuple.

- `max_wt_norm = Inf`: the maximum permissible norm of the weight vector. This
  norm has a big impact on the size of the aliasing bias, so it can sometimes be
  advantageous to reduce `Ω` in exchange for a smaller norm. If this is set to a
  finite value and the computed weights exceed it, then `Ω` is reduced as `Ω .*=
  reduction_factor` and the weights are recomputed.

- `reduction_factor = 0.9`: this quantity determines how aggressively we shrink
  `Ω` in the case when the computed weights exceed `max_wt_norm`. Making it larger
  may save you on the cost of recomputing weights, but it may mean your ultimate
  `Ω` is lower than it could have been.

- `verbose = true`: whether or not to print output.
"""
function window_quadrature_weights(pts, g; solver=default_solver(pts),
                                   Ω=default_Ω(pts, g), max_wt_norm=Inf, 
                                   min_Ω=0.05.*Ω, reduction_factor=0.9, verbose=true)
  wts    = solve_linsys(pts, g, Ω, solver, verbose=verbose)
  verbose && @printf "||α||₂:             %1.5e\n" norm(wts)
  while norm(wts) > max_wt_norm
    Ω    .*= reduction_factor
    all(Ω .< min_Ω) && throw(error("Could not achieve ||α||₂ < $max_wt_norm for Ω .> $min_Ω."))
    wts    = solve_linsys(pts, g, Ω, solver, verbose=verbose)
    verbose && @printf "||α||₂:             %1.5e\n" norm(wts)
  end
  (Ω, wts)
end

struct SpectralDensityEstimator{O,F,W}
  Ω::O
  win::W
  freq::Vector{F}
  sdf::Vector{Float64}
  wts::Vector{ComplexF64}
end

"""
estimate_sdf(pts::Vector{Float64}, data, g; Ω=default_Ω(pts, g),
             frequencies=default_frequencies(pts, g, Ω),
             wts=nothing, kwargs...) -> (frequencies, estimates)

Estimates the spectral density for the stationary process sampled at locations
`pts` and with values given by `data` at frequencies `frequencies`. Each column
in `data` is treated as an iid sample of the process and averaged in the final
estimator. `Ω` is the resolution maximum for the window quadrature weights. 

Any provided keyword arguments are passed to `window_quadrature_weights` if
`wts` is `nothing`, and ignored if `wts` is provided. See the docstrings for
`window_quadrature_weights` for details on available kwargs. If `wts` is
provided, the `Ω` arg also does nothing.
"""
function estimate_sdf(pts, data, g; Ω=default_Ω(pts, g), wts=nothing,
                      frequencies=default_frequencies(pts, g, Ω), kwargs...)
  if isnothing(wts)
    (Ω, wts) = window_quadrature_weights(pts, g; Ω=Ω, kwargs...)
  end
  fs  = NUFFT3(pts, collect(frequencies.*(2*pi)), true, 1e-15)
  out = zeros(ComplexF64, length(frequencies), size(data, 2))
  mul!(out, fs, complex(Diagonal(wts)*data))
  SpectralDensityEstimator(Ω, g, collect(frequencies), 
                           mean(x->abs2.(x), eachcol(out)), wts)
end

