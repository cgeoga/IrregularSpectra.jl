
function default_range(g)
  (a, b) = window_support(g)
  (b-a)/50
end

"""
(weights, fmax) = matern_frequency_selector(pts::Vector{Float64}, g::ClosedFormWindow;
                                            smoothness=1.5, rho=default_range(g),
                                            alias_tol=1.0, Ω=default_Ω(pts, g),
                                            max_wt_norm=0.15, min_fmax=0.05*Ω, 
                                            reduction_factor=0.9)

This function is a **heuristic** tool for estimating the highest possible
frequency for which an SDF can be estimated in such a way that

|est - truth|/truth < alias_tol.

What this function actually does is compare the variance of the linear contrast
of a Matern process at locations `pts` corresponding to the frequency `fmax` and
its intended variance, which is the SDF convolved with the spectral window. If
those numbers disagree in relative tolerance by more than `alias_tol`, then
`fmax` is reduced and the comparison is done again. This is made fast by using a
Vecchia-type approximation for the precision of the Matern process.

Additionally, if the norm of the weights is above `max_wt_norm`, then Ω is
reduced by `reduction_factor` and the weights are recomputed as well.

**NOTE**: for different `smoothness` values, and to a lesser degree range
parameter values (`rho`), the corresponding `fmax` will change!  This
corresponds with the fact that if your SDF decays more slowly, you will have
more aliasing bias to deal with.  So you should pick the smoothness that you
think most closely corresponds to the decay rate of your actual data's SDF. 

**NOTE**: this function requires scalar evaluations of the Fourier transform of
the window function. As such, the window you provide must be a `::ClosedFormWindow`.

**See also**: the docs for `window_quadrature_weights` outline the keyword args in detail.
"""
function matern_frequency_selector(pts, g::ClosedFormWindow; smoothness=1.5, 
                                   rho=default_range(g), alias_tol=1.0, 
                                   Ω=default_Ω(pts, g), min_fmax=0.05*Ω, 
                                   max_wt_norm=Inf, reduction_factor=0.9, verbose=true)
  # finally, we make the actual covariance function.
  kfn  = (x,y) -> matern_cov(abs(x-y), (1.0, rho, smoothness))
  sdf  = w -> matern_sdf(w, (1.0, rho, smoothness))
  g_ft = w -> fouriertransform(g, w)
  # Next, we directly build the sparse Cholesky factor of the precision matrix,
  # getting a matrix such that Sigma ≈ inv(L'*inv(D)*L)).
  (L, D) = sparse_rchol(kfn, pts)
  # Now, we pick an ambitious wmax and check for aliasing control, and in the
  # case where we don't have control we shrink wmax by a factor of alpha.
  bw   = bandwidth(g)
  wts  = window_quadrature_weights(pts, g; Ω=Ω, verbose=false)[2]
  verbose && println("||α||₂ = ", norm(wts))
  while norm(wts) > max_wt_norm
    Ω   *= reduction_factor
    Ω    < min_fmax && throw(error("Could not achieve ||α||_2 < $max_wt_norm for Ω > $min_fmax."))
    wts  = window_quadrature_weights(pts, g; Ω=Ω, verbose=false)[2]
    verbose && println("||α||₂ = ", norm(wts))
  end
  Dw   = Diagonal(wts)
  fmax = 0.9*Ω
  while fmax > min_fmax
    # The highest frequency column of the NUDFT matrix, corresponding to the
    # estimate most likely to have unacceptable aliasing bias.
    fmax_col = [cispi(-2*xj*fmax) for xj in pts]
    # What the expectation of our estimator should be (the true SDF convolved
    # with the window function):
    shouldbe = quadgk(w->abs2(g_ft(w))*sdf(w+fmax), -bw, bw, atol=1e-12)[1]
    # The actual variance of the linear contrast, accelerated by the sparse
    # precision of a Markov process.
    actual   = real(dot(fmax_col, Dw*(L\(D*(L'\(Dw'*fmax_col))))))
    # Compare the error with the aliasing tol, and either return the weights and
    # permissible fmax or reduce fmax and repeat.
    @show (shouldbe, actual)
    abs(actual - shouldbe) < shouldbe*alias_tol && return (wts, fmax)
    verbose && println("Reducing fmax ($fmax)...")
    fmax *= reduction_factor
  end
  # in the case where even fmax=min_fmax did not control aliasing bias to the
  # requested degree, return a NaN for fmax so that this function can't fail in
  # a way that somebody could miss.
  (wts, NaN)
end

