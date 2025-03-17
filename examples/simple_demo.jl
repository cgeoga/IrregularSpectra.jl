
using Printf, IrregularSpectra

# Generate n uniform points on [a, b]:
n      = 3000
(a, b) = (0.0, 1.0)
pts    = sort(rand(n).*(b-a) .+ a)

# Simulate a Matern process at those points:
sim = let kernel = (x,y)->IrregularSpectra.matern_cov(x-y, (1.0, 0.1, 0.75))
  IrregularSpectra.simulate_process(pts, kernel)
end

# Compute the estimator, which we'll do at just a few points for demonstration:
nyquist   = n/(4*(b-a))
est_freqs = range(0.0, 0.4*nyquist, length=10)
window    = Kaiser(20.0, a=a, b=b)
est       = estimate_sdf(pts, sim, window, est_freqs)

# Brief inspection of the output:
@printf "\n    ω        true S(ω)      est. S(ω)\n"
@printf "-------------------------------------\n"
for (j, freq) in enumerate(est_freqs)
  true_sdf = IrregularSpectra.matern_sdf(freq, (1.0, 0.1, 0.75)) 
  @printf "%1.3e    %1.3e      %1.3e\n" freq  true_sdf est[j]
end

# If you want to plot:
many_est_freqs = range(0.0, 0.4*nyquist, length=n)
est       = estimate_sdf(pts, sim, window, many_est_freqs)
tru       = IrregularSpectra.matern_sdf.(many_est_freqs, Ref((1.0, 0.1, 0.75)))
gplot(many_est_freqs, est, tru, ylog=true)

