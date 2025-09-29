
module IrregularSpectraArnoldiMethodExt

  using LinearAlgebra, IrregularSpectra, ArnoldiMethod
  using IrregularSpectra.StaticArrays
  using IrregularSpectra.BandlimitedOperators

  import IrregularSpectra: gridded_nyquist_gpss, _fast_prolate_fromrule

  function gridded_nyquist_gpss(times::Vector{Float64}, bw; concentration_tol=1e-8, 
                                max_tapers=5, min_krylov=max_tapers, 
                                max_krylov=5*max_tapers)
    # Step 1: using a fast sinc transform and ArnoldiMethod.jl, compute the GPSS
    # vectors. Note that especially for wider bandwidths, you will be getting a
    # soup of the well-concentrated vectors because those eigenvalues can differ
    # by, like, eps() from each other. So asking an iterative method to separate
    # those from each other is not in the cards.
    fs  = FastBandlimited(times, times, ω->1.0, bw)
    (res, status) = partialschur(fs; tol=1e-12, nev=max_tapers, 
                                 mindim=min_krylov, maxdim=max_krylov)
    status.converged || throw(error("Partial Schur method failed to converge! Try changing the kwarg `min_krylov` from its default value of `max_tapers` (whose default is `5`)."))
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

  function spatial_area(locations::Vector{SVector{2,Float64}})
    (x_q1, x_qn) = extrema(x->x[1], locations)
    (y_q1, y_qn) = extrema(x->x[2], locations)
    (x_qn - x_q1)*(y_qn - y_q1)
  end

  function get_nyquist_j(locations::Vector{SVector{2,Float64}}, j)
    xj  = getindex.(locations, 1)
    xjs = sort(unique(xj))
    inv(xjs[2] - xjs[1])/2
  end

  function gridded_nyquist_gpss(locations::Vector{SVector{2,Float64}}, bw;
                                concentration_tol=1e-8, max_tapers=5, 
                                min_krylov=max_tapers, 
                                max_krylov=5*max_tapers)
    fs  = FastBandlimited(locations, locations, Ω->1.0, bw; polar=true)
    (res, status) = partialschur(fs; tol=1e-12, nev=max_tapers, 
                                 mindim=min_krylov, maxdim=max_krylov)
    status.converged || throw(error("Partial Schur method failed to converge! Try changing the kwarg `min_krylov` from its default value of `max_tapers` (whose default is `5`)."))
    rel_concs    = real(res.eigenvalues)
    rel_concs  ./= rel_concs[1]
    good_conc_ix = findlast(>=(1-concentration_tol), rel_concs)
    gpss = res.Q[:,1:good_conc_ix]
    # Step 2: L^2 normalize them. This is _not_ the exact normalization scheme
    # suggested by Bronez, and at some point this code could be modified to
    # offer that scheme as well. But this normalization and then flat weighting
    # certainly is not wrong, and it fits into the existing code base much more
    # easily.
    Ω1 = get_nyquist_j(locations, 1)
    Ω2 = get_nyquist_j(locations, 2)
    Ω  = (Ω1, Ω2)
    nq = 4*Int(ceil(sqrt(length(locations))))
    (wgrid, glwts) = IrregularSpectra.glquadrule((nq, nq), .-Ω, Ω)
    F   = IrregularSpectra.NUFFT3(wgrid.*(2*pi), locations, -1)
    tmp = zeros(ComplexF64, length(wgrid))
    for j in 1:size(gpss, 2)
      wtsj = complex(gpss[:,j])
      mul!(tmp, F, wtsj)
      l2norm = sqrt(dot(glwts, abs2.(tmp)))
      view(gpss, :, j) ./= l2norm
    end
    gpss
  end

  struct ConjugatedHermOperator{B}
    D::Diagonal{Float64, Vector{Float64}}
    M::B
  end

  Base.size(co::ConjugatedHermOperator{B})    where{B} = size(co.D)
  Base.size(co::ConjugatedHermOperator{B}, j) where{B} = size(co.D, j)
  Base.eltype(co::ConjugatedHermOperator{B})  where{B} = Float64 # hard-coded for now

  LinearAlgebra.issymmetric(co::ConjugatedHermOperator{B}) where{B} = true
  LinearAlgebra.ishermitian(co::ConjugatedHermOperator{B}) where{B} = true
  function LinearAlgebra.adjoint(co::ConjugatedHermOperator{B}) where{B}
    Adjoint{Float64, ConjugatedHermOperator{B}}(co)
  end

  function LinearAlgebra.mul!(buf, co::ConjugatedHermOperator{B}, v) where{B}
    mul!(buf, co.M, co.D*v)
    D = co.D
    for j in 1:size(D, 1)
      view(buf, j, :) .*= D[j,j]
    end
    buf
  end

  function LinearAlgebra.mul!(buf, co::Adjoint{Float64, ConjugatedHermOperator{B}}, 
                              v) where{B}
    mul!(buf, co.parent, v)
  end

  function _fast_prolate_fromrule(w, nodes, weights; concentration_tol=1e-8,
                                  max_tapers=5, min_krylov=max_tapers, 
                                  max_krylov=5*max_tapers)
    _M = IrregularSpectra.fast_slepian_operator(nodes, nodes, w)
    Dw = Diagonal(sqrt.(weights))
    M  = ConjugatedHermOperator(Dw, _M)
    (res, status) = partialschur(M; tol=1e-12, nev=max_tapers, 
                                 mindim=min_krylov, maxdim=max_krylov)
    status.converged || throw(error("Partial Schur method failed to converge! Try changing the kwarg `min_krylov` from its default value of `max_tapers` (whose default is `5`)."))
    rel_concs    = real(res.eigenvalues)
    rel_concs  ./= rel_concs[1]
    good_conc_ix = findlast(>=(1-concentration_tol), rel_concs)
    prolates = res.Q[:,1:good_conc_ix]
    ldiv!(Dw, prolates)
    for sj in eachcol(prolates)
      sj ./= sqrt(dot(weights, abs2.(sj))) 
    end
    prolates 
  end

end

