
using IrregularSpectra, StaticArrays, Printf
using NearestNeighbors # bring in this weakdep for the large-data methods

# Kernel included in the package for convenience.
kernel(x,y) = IrregularSpectra.matern_cov(x-y, (1.0, 0.05, 1.75))

# Generate n uniform points on [0,1]:
n    = 5000
pts  = rand(SVector{2,Float64}, n)
sims = IrregularSpectra.simulate_process(pts, kernel, 500)

# Compute the estimator, which we'll do at just a few points for demonstration:
bw     = shannon_bandwidth(64.0, (0.0, 0.0), (1.0, 1.0)) # should give ~3 good tapers with defaults
window = Prolate2D(bw,           # (half-)bandwidth 
                   (0.0, 0.0),   # lower bounds of rectangle
                   (1.0, 1.0))   # upper bounds of rectangle
solver = KrylovSolver(SparsePreconditioner(1e-12))
est    = estimate_sdf(pts, sims, window; Ω=(20.0, 20.0))

# Unlike in the 1D case, the window-induced bias in 2+D can be very strong. So
# you shouldn't expect the mean of even many many samples to be close to the
# true SDF. 
@printf "\nNOTE: be mindful that the 2D-based window artifacts are much more extreme."
@printf "\n\nEstimator summary using 500 replicates:\n"
@printf "\n    ||ω||        true S(ω)      est. S(ω)\n"
@printf "-------------------------------------\n"
p = sortperm(est.freq, by=w->norm(w)) # sorting results by norm for quick print
m = length(est.freq)
(freqs_p, ests_p) = (est.freq[p], est.sdf[p])
for j in 1:div(m,10):m 
  fj       = freqs_p[j]
  true_sdf = IrregularSpectra.matern_sdf(fj, (1.0, 0.05, 1.25)) 
  @printf "%1.3e    %1.3e      %1.3e\n" norm(fj)  true_sdf ests_p[j]
end

