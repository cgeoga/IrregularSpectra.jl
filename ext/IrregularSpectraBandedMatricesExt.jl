
module IrregularSpectraBandedMatricesExt

  using IrregularSpectra, NearestNeighbors, BandedMatrices
  using IrregularSpectra.Printf
  using IrregularSpectra.LinearAlgebra
  using NearestNeighbors.StaticArrays
  import IrregularSpectra: BandedPreconditioner, gen_kernel, krylov_preconditioner!

  # Normalized to be supported on [-1, 1].
  struct UnitBSpline{N} end

  @generated function (ubs::UnitBSpline{N})(x::T) where{N,T}
    (Nm1, Np1) = (N-1, N+1)
    factNm1    = factorial(Nm1)
    normalizer = T((1, 1, 3/4, 2/3, 115/192, 11/20, 5887/11520)[N]) # N <= 6 for now...
    divisor = factNm1*normalizer
    quote
      abs(x) >= 1.0 && return zero($T)
      _x = (x + 1)*$N/2
      isone($N) && return $T(zero($T) <= _x <= one($T))
      (sgn, out) = (one($T), zero($T))
      Base.Cartesian.@nexprs $Np1 j -> begin
        k    = j-1
        term = sgn*binomial($N, k)*max(_x - k, zero($T))^($Nm1)
        out += term
        sgn  = -sgn
      end
      out/$divisor
    end
  end

  function krylov_preconditioner!(pts_sa::Vector{SVector{D,Float64}}, Ω, 
                                  solver::KrylovSolver{BandedPreconditioner, K};
                                  verbose=true) where{D,K}
    kernel = gen_kernel(solver, pts_sa, Ω)
    pre_time = @elapsed begin
      # reorder the points with a KD-tree to make the banded approximation at
      # least slightly more accurate (if we aren't in 1D, in which case we
      # assume the points are sorted already).
      if D > 1
        tree      = KDTree(pts_sa)
        tree_data = tree.data
        foreach(j->(pts_sa[j] = tree_data[j]), eachindex(pts_sa)) # permute the points
      end
      # make the banded matrix:
      n         = length(pts_sa)
      bw        = solver.preconditioner.bandwidth
      # TODO (cg 2025/07/01 14:02): This taper is just killing the accuracy
      # here. It would be interesting to try and cook up one that was flatter
      # around the origin.
      taper     = UnitBSpline{6}()
      diag_vals = [taper(abs(h)/bw) for h in 0:(bw)]
      bm        = BandedMatrix{Float64}(undef, (n, n), (bw, bw))
      for j in 1:n
        nzixs = max(1, j-bw):min(n, j+bw)
        for k in nzixs
          offset  = abs(j-k)+1
          bm[j,k] = diag_vals[offset]*kernel(pts_sa[j], pts_sa[k])
          bm[k,j] = bm[j,k]
        end
        bm[j,j] += solver.perturbation
      end
      bmf = cholesky!(Symmetric(bm))
    end
    verbose && @printf "preconditioner assembly time: %1.3fs\n" pre_time
    (true, bmf)
  end

end

