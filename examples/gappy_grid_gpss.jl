
using IrregularSpectra

# Points on a gappy lattice (for example):
full_lattice  = range(0.0, 1.0, length=5_000)
gappy_lattice = full_lattice[vcat(1:1000, 2000:4000, 4100:5000)]

# A single replicate...
kernel(x, y) = IrregularSpectra.matern_cov(x-y, (1.0, 0.05, 1.25))
sims = IrregularSpectra.simulate_process(gappy_lattice, kernel, 1)

# ...but a big bandwidth and a slightly lax concentration tolerance.
gpss        = GPSS(gappy_lattice, 15.0; concentration_tol=1e-5)
frequencies = range(0.0, 1500.0, length=5_000)
estimator   = estimate_sdf(gappy_lattice, sims, gpss; frequencies)

# However you like to make your plots!
#=
plot(estimator.freq, estimator.sdf, 
    IrregularSpectra.matern_sdf.(frequencies, Ref((1.0, 0.05, 1.25))),
    ylog=true)
=#

