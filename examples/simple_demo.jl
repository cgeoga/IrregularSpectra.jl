
using Printf, IrregularSpectra

# Generate n uniform points on [a, b]:
n      = 3000
(a, b) = (-1.0, 1.0)
pts    = sort(rand(n).*(b-a) .+ a)

# Simulate a Matern process at those points using m replicates:
m    = 500
sims = let kernel = (x,y)->IrregularSpectra.matern_cov(x-y, (1.0, 0.05, 1.75))
  IrregularSpectra.simulate_process(pts, kernel, m)
end

# Compute the estimator, which we'll do at just a few points for demonstration:
Ω         = 0.5*n/(4*(b-a))
est_freqs = range(0.0, Ω/2, length=30)
window    = Kaiser(6.0, a=a, b=b)
est       = estimate_sdf(pts, sims, window, est_freqs; Ω=Ω)

# Brief inspection of the output:
@printf "\n\nEstimator summary using m=%i replicates:\n" m
@printf "\n    ω        true S(ω)      est. S(ω)\n"
@printf "-------------------------------------\n"
for (j, freq) in enumerate(est_freqs)
  true_sdf = IrregularSpectra.matern_sdf(freq, (1.0, 0.05, 1.75)) 
  @printf "%1.3e    %1.3e      %1.3e\n" freq  true_sdf est[j]
end

