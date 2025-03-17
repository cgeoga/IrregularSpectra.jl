
# IrregularSpectra.jl

This repository is the software library companion to [Nonparametric spectral
density estimation from irregularly sampled data](https://arxiv.org/abs/2503.00492).

# Basic usage demonstration

Here is a heavily commented demonstration, which can also be found as a plain
code file in `./examples/simple_demo.jl`.

First, let's generate some data. This picks `n=3000` uniform locations on `[a,
b] = [0, 1]` and simulates a Matern process with the chosen parameters at those
values. This package offers the Matern kernel and spectral density both for
convenience and for use in some upcoming heuristic tools for selecting the
maximum resolved frequency Ω, which is a central concept in the paper and
directly controls the tradeoff between how far into the tails of the SDF you can
estimate and the size of the bias.
```{julia}
using IrregularSpectra

# Generate n uniform points on [a, b]:
n      = 3000
(a, b) = (0.0, 1.0)
pts    = sort(rand(n).*(b-a) .+ a)

# Simulate a Matern process at those points:
sim = let kernel = (x,y)->IrregularSpectra.matern_cov(x-y, (1.0, 0.1, 0.75))
  IrregularSpectra.simulate_process(pts, kernel)
end
```

From here, the estimator is easy to obtain: we simply pick the frequencies at
which we'd like to estimate the SDF, select the window function we wish to use
(and it is easy to bring your own!), and use the `estimate_sdf` function. The
choice `20.0` is the shape parameter with the Kaiser window, and for reference
on the domain of `[a, b] = [0, 1]` that gives a main lobe half-bandwidth of
`20/pi`, which is about `6.3`.
```{julia}
nyquist   = n/(4*(b-a))
est_freqs = range(0.0, 0.4*nyquist, length=10)
window    = Kaiser(20.0, a=a, b=b)
est       = estimate_sdf(pts, sim, window, est_freqs)
```
**Note:** we are still finalizing the design interface for prolate functions,
which can provide significantly more performant weights for points sampled on
domains with large gaps (see the paper for demonstrations). If you directly
translate this demo code to point samples with big gaps, you will get weights
with a very large norm and all of your estimates will be drowned out with bias
from unresolvable frequencies. So this demo code only applies for sampling
schemes that don't involve gaps or other pathologies where the maximum point gap
doesn't go to zero. Please stay tuned for a more general interface (which is
coming) and open issues as you experience them, because it is likely that we can
suggest better window functions even if they are not currently implemented in
this package.


The script document in `./examples/simple_demo.jl` just uses printing as a
diagnostic, but here is a visual one where we estimate at more frequencies (this
plot was made with my own wrapper of Gnuplot that uses sixel output, but
substitute with your preferred plotting tool):
```{julia}
many_est_freqs = range(0.0, 0.4*nyquist, length=n)
est   = estimate_sdf(pts, sim, window, many_est_freqs)
truth = IrregularSpectra.matern_sdf.(many_est_freqs, Ref((1.0, 0.1, 0.75)))
gplot(many_est_freqs, est, truth, ylog=true)
```
<p align="center">
    <img src="quicksixel_est_demo.png" alt="A sample estimator plot" width=600>
</p>


# Roadmap

This software library is under very active development. An incomplete list of
features to expect in the near future:

- A helper function to use a Matern process to automatically select a suitable
  Ω. This is implemented and in `./src/` now, but not exported or documented
  as the exact interface design is still being discussed.
- A heuristic tool for handling potentially gappy one-dimensional domains that
  adaptively splits the data domain into disjoint segments based on large gaps
  and balancing the tradeoff between reducing the norm of the weights and the
  off-lobe power in the spectral window. This is again something we have
  implemented and all that is left to do is to polish it.
- An interface for providing gappy one-dimensional domain information and
  obtaining prolate function evaluations and right-hand sides for weight
  calculation. This is again done already---the code just needs to be done and
  the interface polished.
- An interface for providing arbitrary points in arbitrary dimensions and
  obtaining prolate function evaluations and right-hand sides for weight
  calculation. This is done _except_ for the step of a designing a robust function
  for automatically obtaining a quadrature rule on the point domain given just
  points. But good tools exist for triangulation-based methods, and so we just
  need to hook into them. Contributions in this space would certainly be welcome.

