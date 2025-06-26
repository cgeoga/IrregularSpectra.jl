
module IrregularSpectraArnoldiMethodExt

  using LinearAlgebra, IrregularSpectra, ArnoldiMethod
  using IrregularSpectra.BandlimitedOperators

  import IrregularSpectra: gridded_nyquist_gpss

  function gridded_nyquist_gpss(times::Vector{Float64}, bw; concentration_tol=1e-4)
    # Step 1: using a fast sinc transform and ArnoldiMethod.jl, compute the GPSS
    # vectors. Note that especially for wider bandwidths, you will be getting a
    # soup of the well-concentrated vectors because those eigenvalues can differ
    # by, like, eps() from each other. So asking an iterative method to separate
    # those from each other is not in the cards.
    fs  = FastBandlimited(times, times, ω->inv(2*bw), bw)
    nev = Int(ceil(bw*(times[end]-times[1])))
    (res, status) = partialschur(fs; tol=1e-12, nev=nev)
    status.converged || throw(error("Partial Schur method failed to converge! Please simply try again, and if the error continues to happen reduce the bandwidth."))
    rel_concs    = real(res.eigenvalues)
    rel_concs  ./= rel_concs[1]
    good_conc_ix = findlast(>=(1-concentration_tol), rel_concs)
    gpss = res.Q[:,1:good_conc_ix]
    # Step 2: L^2 normalize them. This is _not_ the exact normalization scheme
    # suggested by Bronez, and at some point this code could be modified to
    # offer that scheme as well. But this normalization and then flat weighting
    # certainly is not wrong, and it fits into the existing code base much more
    # easily.
    Ω   = inv(minimum(abs, diff(times)))/2
    (wgrid, glwts) = IrregularSpectra.glquadrule(4*length(times), -Ω, Ω)
    F   = IrregularSpectra.NUFFT3(wgrid.*(2*pi), times, -1)
    tmp = zeros(ComplexF64, length(wgrid))
    for j in 1:size(gpss, 2)
      wtsj = complex(gpss[:,j])
      mul!(tmp, F, wtsj)
      l2norm = sqrt(dot(glwts, abs2.(tmp)))
      view(gpss, :, j) ./= l2norm
    end
    gpss
  end

end

