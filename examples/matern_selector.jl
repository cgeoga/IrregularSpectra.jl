
using Printf, IrregularSpectra

# Generate n uniform points on [a, b]:
n      = 3000
(a, b) = (-1.0, 1.0)
pts    = sort(rand(n).*(b-a) .+ a)

# Simulate a Matern process at those points using m replicates. this time, we
# pick a process 
m    = 500
sims = let kernel = (x,y)->IrregularSpectra.matern_cov(x-y, (1.0, 0.1, 0.65))
  IrregularSpectra.simulate_process(pts, kernel, m)
end

# Compute the estimator, which we'll do at just a few points for demonstration:
window      = Kaiser(6.0, a=a, b=b)
(wts, fmax) = matern_frequency_selector(pts, window, smoothness=0.5, alias_tol=0.1)
est_freqs   = range(0.0, fmax, length=30)
est         = estimate_sdf(pts, sims, window, est_freqs; wts=wts)

# Brief inspection of the output:
@printf "\n\nEstimator summary using m=%i replicates:\n" m
@printf "\n    ω        true S(ω)      est. S(ω)\n"
@printf "-------------------------------------\n"
for (j, freq) in enumerate(est_freqs)
  true_sdf = IrregularSpectra.matern_sdf(freq, (1.0, 0.1, 0.65)) 
  @printf "%1.3e    %1.3e      %1.3e\n" freq  true_sdf est[j]
end

