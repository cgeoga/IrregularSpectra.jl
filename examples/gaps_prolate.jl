
using Printf, IrregularSpectra

# Matern kernel included for convenience.
kernel(x,y) = IrregularSpectra.matern_cov(x-y, (1.0, 0.05, 1.75))

# Generate n1 uniform points on (a1, b1) and n2 uniform points in (a2, b2):
(a1, b1, n1) = (0.0, 1.0, 1000)
(a2, b2, n2) = (1.5, 3.5, 1500)
pts = sort(vcat(rand(n1).*(b1-a1) .+ a1, rand(n2).*(b2-a2) .+ a2))

# Simulate a Matern process at those points using m replicates. 
sims = IrregularSpectra.simulate_process(pts, kernel, 500)

# Compute the estimator (compare with simple_demo.jl: just one more line of code
# here, and the rest of the details are handled by hopefully sane defaults).
intervals = gappy_intervals(pts)
window    = Prolate1D(intervals)
est       = estimate_sdf(pts, sims, window)

# Brief inspection of the output:
@printf "\n\nEstimator summary using m=500 replicates:\n"
@printf "\n    ω        true S(ω)      est. S(ω)\n"
@printf "-------------------------------------\n"
m = length(est.freq)
for j in 1:div(m,10):m
  fj       = est.freq[j]
  true_sdf = IrregularSpectra.matern_sdf(fj, (1.0, 0.05, 1.75)) 
  @printf "%1.3e    %1.3e      %1.3e\n" fj  true_sdf est.sdf[j]
end

