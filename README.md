
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
```julia
using IrregularSpectra

# Generate n uniform points on [a, b]:
n      = 3000
(a, b) = (-1.0, 1.0)
pts    = sort(rand(n).*(b-a) .+ a)

# Simulate a Matern process at those points with m replicates:
m   = 500 # number of replicates
sim = let kernel = (x,y)->IrregularSpectra.matern_cov(x-y, (1.0, 0.1, 1.75))
  IrregularSpectra.simulate_process(pts, kernel, m)
end
```

From here, the estimator is easy to obtain: we simply pick the frequencies at
which we'd like to estimate the SDF, select the window function we wish to use
(and it is easy to bring your own!), and use the `estimate_sdf` function. The
choice `20.0` is the shape parameter with the Kaiser window, and for reference
on the domain of `[a, b] = [-1, 1]` that gives a main lobe half-bandwidth of
`20/(2*pi)`, which is about `3.2`. The function `estimate_sdf`, if given
multiple iid samples as columns in `sims`, will average the multiple estimates.
```julia
Ω         = 0.5*n/(4*(b-a)) # half of theoretical nyquist max
est_freqs = range(0.0, Ω/2, length=1000)
window    = Kaiser(20.0, a=a, b=b)
est       = estimate_sdf(pts, sims, window, est_freqs; Ω=Ω)
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
```julia
truth = IrregularSpectra.matern_sdf.(many_est_freqs, Ref((1.0, 0.1, 1.75)))
est1  = estimate_sdf(pts, sims[:,1], window, est_freqs; Ω=Ω) 
gplot(est_freqs, est1, est, truth, ylog=true)
```

<p align="center">
    <img src="quicksixel_est_demo.png" alt="A sample estimator plot" width=600>
</p>

This plot shows the estimator from a single sample (purple), the average
estimate from `m=500` replicates is shown in blue (indicating that the bias in
the estimate is minimal compared to its expected value), and the true SDF is
shown in green.

# Experimental features/interfaces

The rate at which aliasing bias can dominate an estimate for the SDF depends
strongly on several factors. The primary two factors are the norm of the weights
and the rate at which the true SDF decays. Naturally in real applications of
nonparametric estimation we don't have access to the rate at which the SDF
decays. But a heuristic tool for at least beginning to make the decisions of
selecting Ω and the highest frequency to estimate in a structured way is offered
in `matern_frequency_selector`. This function exactly computes the cutoff
frequency for which the bias of an SDF estimator exceeds a relative tolerance of
the true SDF for a Matern process with user-selectable smoothness and range
parameters. Even for very large `n`, the weight vector computation is by far the
most expensive component since this code internally uses sparse precision
approximations to accelerate all covariance matrix operations.

In action, you might modify the above code to do this instead:
```julia
(wts, fmax) = matern_frequency_selector(pts, window, smoothness=0.5, alias_tol=0.1)
est_freqs   = range(0.0, fmax, length=1000)
est         = estimate_sdf(pts, sims, window, est_freqs; wts=wts)
```
In particular:
- The `smoothness` argument is the smoothness used in the Matern process. The
  lower this is, the more conservative your `fmax` will be.
- There is also a kwarg `rho`, for the range parameter. This code automatically
  selects a reasonable default, but you are welcome to change it.
- `alias_tol` is the cutoff used to determine how much `fmax` needs to be
  reduced. In particular, `fmax` is reduced until `|S(fmax) - \E
  \hat{S}(fmax)|/S(fmax) < alias_tol`. The smaller this value is, the smaller
  `fmax` will be---but the smaller the relative bias from aliasing will be as well.
  The choice of this quantity depends on your needs.
- Finally, there is also a `max_wt_norm` argument, with default value `0.15`.
  Sometimes you can find a safe way to look deeper into a spectrum by reducing
  Ω, which will reduce the norm of the weights, and in turn reduce the magnitude
  of the aliasing bias for every estimate. At this time, we do not have an easy
  answer about when reducing Ω can actually lead to a higher `fmax`. But
  especially for non-differentiable processes it clearly can happen.

**NOTE:** Again, this interface will fail to give you good output if your window
function is not well-selected for your problem. If you have large gaps, please
wait for the prolate window interface. We'll get it sorted out and available
here as soon as possible!

This alternative estimator workflow is demonstrated as a plain code file in
`./examples/matern_selector.jl`.


# Roadmap

This software library is under very active development. An incomplete list of
features to expect in the near future:

- A heuristic tool for handling potentially gappy one-dimensional domains that
  adaptively splits the data domain into disjoint segments based on large gaps
  and balancing the tradeoff between reducing the norm of the weights and the
  off-lobe power in the spectral window. This is again something we have
  implemented and all that is left to do is to polish it.
- An interface for providing arbitrary points in arbitrary dimensions and
  obtaining prolate function evaluations and right-hand sides for weight
  calculation. This is done _except_ for the step of a designing a robust function
  for automatically obtaining a quadrature rule on the point domain given just
  points. But good tools exist for triangulation-based methods, and so we just
  need to hook into them. Contributions in this space would certainly be welcome.
- A tool for classifying different categories of sampling schemes. Samples on a
  regular grid can be handled much faster, and same for a gappy regular grid
  (although with different mechanisms for each). It would be very nice for this
  tool to be sufficiently general that it takes _any_ points and gives you back
  a decent estimator that has been computed as rapidly as possible.
- A tool for classifying different varieties of gappiness. As an extreme
  example, consider you have n points on [0, 0.01] and [10, 100]. In the first
  interval, you will be able to resolve exceptionally high frequencies, but your
  bandwidth will have to be on the order of 100 or 1000, and so you will be
  unlikely to resolve anything in the low frequency range. The situation will be
  entirely reversed on the second interval. It does not make sense to compute one
  set of weights jointly for both segments, and it would be much better to use the
  two different sections of data to estimate different parts of the SDF. It would
  be interesting to implement some variety of `CompositeEstimator` type that
  handles that decision making at least somewhat automatically.
- It would be nice to stabilize some kind of result-type API and exported getter
  functions to protect users from changes that really should only be relevant to
  internal functions and developers. Maybe some kind of result type like
  `SpectralEstimator` or something.

