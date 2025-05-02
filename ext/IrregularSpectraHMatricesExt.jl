
module IrregularSpectraHMatricesExt

  using IrregularSpectra, HMatrices
  using IrregularSpectra.LinearAlgebra
  using IrregularSpectra.Printf

  using IrregularSpectra: KrylovSolver, HMatrixPreconditioner, getdim, glquadrule, linsys_rhs, static_points, NUFFT3

  function has_empty_leaves(H)
    sparse_leaves = filter(x->HMatrices.isadmissible(x), HMatrices.leaves(H))
    (rmin, rmax)  = extrema(x->HMatrices.rank(x.data), sparse_leaves)
    iszero(rmin) 
  end

  function IrregularSpectra.krylov_preconditioner(pts_sa, Ω, solver::KrylovSolver{HMatrixPreconditioner,K}; 
                                                  verbose=false) where{K}
    if length(Ω) > 1 && K == IrregularSpectra.SincKernel
      @warn "In higher dimensions with the H-matrix preconditioner, please use the `GaussKernel` preconditioner kernel and not the `SincKernel` for better performance."
    end
    kernel   = IrregularSpectra.gen_preconditioner_kernel(solver, Ω)
    sk       = KernelMatrix(kernel, pts_sa, pts_sa)
    pre_time = @elapsed begin
      H  = assemble_hmatrix(sk; atol=solver.preconditioner.atol)
      Hf = if has_empty_leaves(H) 
        @warn "The H-matrix preconditioner approximation has empty leaves and cannot be factorized. Falling back to an identity preconditioner..."
        I 
      else
        lu(H; atol=solver.preconditioner.luatol)
      end
    end
    verbose && @printf "Preconditioner assembly time: %1.3fs\n" pre_time
    Hf
  end

end

