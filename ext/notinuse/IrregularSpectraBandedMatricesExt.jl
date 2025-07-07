
module IrregularSpectraBandedMatricesExt

  using IrregularSpectra, BandedMatrices
  using IrregularSpectra.Printf
  using IrregularSpectra.LinearAlgebra
  using IrregularSpectra.StaticArrays
  import IrregularSpectra: BandedPreconditioner, gen_kernel, krylov_preconditioner!

  # TODO (cg 2025/07/07 15:22): write a function that automatically picks the
  # bandwidth.

  function krylov_preconditioner!(pts_sa::Vector{SVector{D,Float64}}, Ω, 
                                  solver::KrylovSolver{BandedPreconditioner, K};
                                  verbose=true) where{D,K}
    kernel = gen_kernel(solver, pts_sa, Ω)
    pre_time = @elapsed begin
      if D > 1
        throw(error("Currently, this BandedMatrix preconditioner is only supported in 1D."))
      end
      n  = length(pts_sa)
      bw = solver.preconditioner.bandwidth
      bm = BandedMatrix{Float64}(undef, (n, n), (bw, bw))
      for j in 1:n
        for k in  max(1, j-bw):min(n, j+bw)
          bm[j,k] = kernel(pts_sa[j], pts_sa[k])
          bm[k,j] = bm[j,k]
        end
      end
      bmf = cholesky!(Symmetric(bm))
    end
    verbose && @printf "preconditioner assembly time: %1.3fs\n" pre_time
    (true, bmf)
  end

end

