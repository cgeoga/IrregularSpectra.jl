
using Printf, IrregularSpectra

# Generate n1 uniform points on (a1, b1) and n2 uniform points in (a2, b2):
(a1, b1, n1) = (0.0, 1.0, 1000)
(a2, b2, n2) = (1.5, 3.5, 1500)
pts = sort(vcat(rand(n1).*(b1-a1) .+ a1, rand(n2).*(b2-a2) .+ a2))

# Simulate a Matern process at those points using m replicates. 
m    = 500
sims = let kernel = (x,y)->IrregularSpectra.matern_cov(x-y, (1.0, 0.05, 1.75))
  IrregularSpectra.simulate_process(pts, kernel, m)
end

# Compute the estimator, which we'll do at just a few points for demonstration.
# Note that this manual selection of fmax and est_freqs is just to show you the
# syntax of providing custom values of those yourself. As the main README and
# docstrings clarify, this package does have (heuristic!) default choices.
intervals   = gappy_intervals(pts)
window      = Prolate1D(intervals)
fmax        = 0.25*min(n1/(4*(b1-a1)), n2/(4*(b2-a2)))
est_freqs   = range(0.0, fmax, length=30)
est         = estimate_sdf(pts, sims, window; frequencies=est_freqs)

# Brief inspection of the output:
@printf "\n\nEstimator summary using m=%i replicates:\n" m
@printf "\n    ω        true S(ω)      est. S(ω)\n"
@printf "-------------------------------------\n"
for (j, freq) in enumerate(est_freqs)
  true_sdf = IrregularSpectra.matern_sdf(freq, (1.0, 0.05, 1.75)) 
  @printf "%1.3e    %1.3e      %1.3e\n" freq  true_sdf est.sdf[j]
end

