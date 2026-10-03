
using Printf, IrregularSpectra
# using HMatrices # if your dataset is large, bring in this weakdep!

# Generate n uniform points on [a, b] and simulate 500 replicates of a Matern
# process at those points. In the real application settings, this would just be
# you loading in your data.
kernel(x, y) = IrregularSpectra.matern_cov(x-y, (1.0, 0.05, 1.25)) # a convenience offering
n      = 3000
(a, b) = (-3.0, 5.0)
pts    = sort(rand(n).*(b-a) .+ a)
sims   = IrregularSpectra.simulate_process(pts, kernel, 500)

# Specify your window function. Newly, we offer the function shannon_bandwidth
# to automatically select an appropriate bandwidth based on your measurement
# locations. A "Shannon bandwidth" of 10 should yield ~3 tapers if other default
# values are used for measurements on a line segment.
bw        = shannon_bandwidth(10.0, a, b)
window    = Prolate1D((a,b); bandwidth=bw)
est       = estimate_sdf(pts, sims, window)

#= Or if your dataset is large, compute est like this instead:
solver    = KrylovSolver(HMatrixPreconditioner(1e-12, 1e-12))
est       = estimate_sdf(pts, sims, window; solver=solver)
=#

# Brief inspection of the output.
#
# NOTE: the default maximum frequency Ω may still be too high for processes with
# slowly decaying S(ω). As you can here, some aliasing bias _may_ start to show
# in the higher frequencies depending on random fluctuations in the norm of the
# weights (see the paper for details!). This isn't something that a
# nonparametric estimator can really account for, so be mindful in your own
# applications!
@printf "\n\nEstimator summary using 500 replicates:\n"
@printf "\n    ω        true S(ω)      est. S(ω)\n"
@printf "-------------------------------------\n"
m = length(est.freq)
for j in 1:div(m,10):m 
  fj       = est.freq[j]
  true_sdf = IrregularSpectra.matern_sdf(fj, (1.0, 0.05, 1.25)) 
  @printf "%1.3e    %1.3e      %1.3e\n" fj  true_sdf est.sdf[j]
end

