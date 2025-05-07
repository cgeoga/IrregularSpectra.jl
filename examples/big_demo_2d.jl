
using IrregularSpectra, StaticArrays, HMatrices

# Generate n uniform points on [0,1]:
n   = 50_000
pts = rand(SVector{2,Float64}, n)

# It is involved to write code to exactly simulate processes for large n and no
# grid structure. So this is just white noise to demonstrate the computational
# workflow. In real applications, each column of sims should be your data.
m    = 500
sims = randn(n, m)

# Compute the estimator, which we'll do at just a few points for demonstration:
window = Prolate2D(4.0,         # (half-)bandwidth
                   (0.0, 0.0),  # lower left  corner of rectangle
                   (1.0, 1.0))  # upper right corner of rectangle
solver = KrylovSolver(HMatrixPreconditioner(1e-9, 1e-9), GaussKernel, # note GaussKernel!
                      perturbation=1e-8) 
est    = estimate_sdf(pts, sims, window; solver=solver)

# Unlike in the 1D case, the window-induced bias in 2+D can be very strong. So
# you shouldn't expect the mean of even many many samples to be close to the
# true SDF. But you can still visualize your result like this:
#=
freq_1dsize = Int(sqrt(length(est.freq)))
heatmap(reshape(log10.(est.sdf), (freq_1dsize, freq_1dsize)))
=#

