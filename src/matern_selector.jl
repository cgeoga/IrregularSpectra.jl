
"""
matern_cov(t, (sigma, rho, smoothness))

The Matern covariance function, specifically parameterized to be the exact
Fourier transform pair of `matern_sdf` (with the 2*pi in the exponential).
"""
function matern_cov(t, p)
  (sig, rho, v) = p
  iszero(t) && return sig^2
  arg = sqrt(2*v)*norm(t)/rho
  (sig^2)*((2^(1-v))/Bessels.gamma(v))*(arg^v)*Bessels.besselk(v, arg)
end

"""
matern_sdf(ω, (sigma, rho, smoothness))

The Matern spectral density function, specifically parameterized to be the exact
Fourier transform pair of `matern_cov` (with the 2*pi in the exponential).
"""
function matern_sdf(w, p) 
  d = length(w)
  (sig, rho, v) = p
  fpi2 = 4*pi^2
  pre  = (2^d)*(pi^(d/2))*Bessels.gamma(v+d/2)*((2*v)^v)/(Bessels.gamma(v)*rho^(2*v))
  (sig^2)*pre*(2*v/(rho^2) + fpi2*norm(w)^2)^(-v - d/2)
end

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
    abs(actual - shouldbe) < shouldbe*alias_tol && return (wts, fmax)
    verbose && println("Reducing fmax ($fmax)...")
    fmax *= reduction_factor
  end
  # in the case where even fmax=min_fmax did not control aliasing bias to the
  # requested degree, return a NaN for fmax so that this function can't fail in
  # a way that somebody could miss.
  (wts, NaN)
end

# This is a very quick-and-dirty function to compute a sparse inverse Cholesky
# factor (or rather, an LDLt factorization with Sigma ≈ inv(L'*inv(D)*L)).  It
# is normally the backbone of Vecchia-type methods, but in this setting it is
# basically exact.
#
# TODO (cg 2024/11/21 11:41): this is my max-simple demonstration of the rchol.
# This could of course be optimized a great deal. Including:
# -- using banded matrices instead of sparse
# -- pre-allocating buffers and parallelizing 
# -- I'm sure other stuff too
# But I doubt that this will _ever_ be the bottleneck even as it is, so it
# doesn't seem like a high priority to me.
function sparse_rchol(kfn, pts)
  n = length(pts)
  L = sparse(Diagonal(ones(n)))
  D = zeros(n)
  for j in 1:n
    cix  = max(1, j-20):(j-1)
    pt   = pts[j]
    cpts = pts[cix]
    Sjj  = kfn(pt, pt)
    if j > 1
      Scc = cholesky!(Hermitian([kfn(x,y) for x in cpts, y in cpts]))
      Scj = [kfn(x, pt) for x in cpts]
      kweights = Scc\Scj
      ccov     = Sjj - dot(Scj, kweights)
      D[j]     = ccov
      view(L, j, cix) .= -kweights
    else
      D[j] = Sjj
    end
  end
  (LowerTriangular(L), Diagonal(D))
end

