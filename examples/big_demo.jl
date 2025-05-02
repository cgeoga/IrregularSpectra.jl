
using Printf, IrregularSpectra, HMatrices

# Generate n uniform points on [a, b]:
n      = 50_000
(a, b) = (-1.0, 1.0)
pts    = sort(rand(n).*(b-a) .+ a)

# Simulate a Matern process at those points using m replicates. But since it is
# big, we have to be a little clever. This uses some internal functions in
# IrregularSpectra.jl to assemble the sparse inverse Cholesky factor of the
# covariance matrix for a Matern(ν=1/2) process, which is precisely Markov.
m    = 500
sims = let kernel = (x,y)->IrregularSpectra.matern_cov(x-y, (1.0, 0.05, 0.5))
  (L, D) = IrregularSpectra.sparse_rchol(kernel, pts; k=2)
  L\(sqrt(D)*randn(length(pts), m))
end

# Compute the estimator, but this time we provide a custom solver via a keyword
# argument. This HMatrixPreconditioner scales very well with data size, and so
# even though we have 50k points and have to solve a dense linear system you get
# your estimates in a few seconds.
window = Kaiser(6.0, a=a, b=b)
solver = KrylovSolver(HMatrixPreconditioner(1e-8, 1e-8), SincKernel, 1e-8)
freqs  = range(0.0, 1000.0, length=20) # just a freq freqs so the print summary below is short
est    = estimate_sdf(pts, sims, window; solver=solver, frequencies=freqs)

# Brief inspection of the output:
@printf "\n\nEstimator summary using m=%i replicates:\n" m
@printf "\n    ω        true S(ω)      est. S(ω)\n"
@printf "-------------------------------------\n"
for k in eachindex(est.freq)
  true_sdf = IrregularSpectra.matern_sdf(est.freq[k], (1.0, 0.05, 0.5)) 
  @printf "%1.3e    %1.3e      %1.3e\n" est.freq[k] true_sdf est.sdf[k]
end

