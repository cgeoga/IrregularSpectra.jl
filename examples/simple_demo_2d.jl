
using IrregularSpectra, StaticArrays

# Generate n uniform points on [0,1]:
n   = 5000
pts = rand(SVector{2,Float64}, n)

# Simulate a Matern process at those points using m replicates:
m    = 500
sims = let kernel = (x,y)->IrregularSpectra.matern_cov(x-y, (1.0, 0.05, 1.75))
  IrregularSpectra.simulate_process(pts, kernel, m)
end

# Compute the estimator, which we'll do at just a few points for demonstration:
window = Prolate2D(4.0,         # (half-)bandwidth 
                   (0.0, 0.0),  # lower bounds of rectangle
                   (1.0, 1.0))  # upper bounds of rectangle
est    = estimate_sdf(pts, sims, window)

# Unlike in the 1D case, the window-induced bias in 2+D can be very strong. So
# you shouldn't expect the mean of even many many samples to be close to the
# true SDF. But you can still visualize your result like this:
#=
freq_1dsize = Int(sqrt(length(est.freq)))
heatmap(reshape(log10.(est.sdf), (freq_1dsize, freq_1dsize)))
=#

