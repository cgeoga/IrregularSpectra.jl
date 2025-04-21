
# This test is adapted from the demonstration file. It just confirms that the
# actual observed mean is within 2x the requested tolerance. While the Matern
# selector is very much a heuristic, it is convenient to test because it also
# hits a lot of other functions.

n      = 1000
(a, b) = (-1.0, 1.0)
pts    = sort(rand(StableRNG(123), n).*(b-a) .+ a)

m    = 500
sims = let kernel = (x,y)->IrregularSpectra.matern_cov(x-y, (1.0, 0.1, 0.65))
  IrregularSpectra.simulate_process(pts, kernel, m; rng=StableRNG(123))
end

window      = Kaiser(6.0, a=a, b=b)
(wts, fmax) = matern_frequency_selector(pts, window, smoothness=0.5, alias_tol=0.1; verbose=false)
est_freqs   = range(0.0, fmax, length=30)
est         = estimate_sdf(pts, sims, window, est_freqs; wts=wts)

win_ft(w)   = IrregularSpectra.fouriertransform(window, w)
sdf(w)      = IrregularSpectra.matern_sdf(w, (1.0, 0.1, 0.65))

bw          = IrregularSpectra.bandwidth(window)
should      = map(est_freqs) do fj
  IrregularSpectra.quadgk(w->abs2(win_ft(w))*sdf(w+fj), -bw, bw, atol=1e-12)[1]
end

err = (est - should)./should
@test maximum(err) < 0.2

