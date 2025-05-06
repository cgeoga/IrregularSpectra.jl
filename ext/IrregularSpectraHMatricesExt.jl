
module IrregularSpectraHMatricesExt

  using IrregularSpectra, HMatrices
  using IrregularSpectra.LinearAlgebra
  using IrregularSpectra.Printf

  using IrregularSpectra: KrylovSolver, HMatrixPreconditioner

  function IrregularSpectra.krylov_preconditioner(pts_sa, Ω, solver::KrylovSolver{HMatrixPreconditioner,K}; 
                                                  verbose=false) where{K}
    if length(Ω) > 1 && K == IrregularSpectra.SincKernel
      @warn "In higher dimensions with the H-matrix preconditioner, please use the `GaussKernel` preconditioner kernel and not the `SincKernel` for better performance."
    end
    kernel   = IrregularSpectra.gen_kernel(solver, pts_sa, Ω)
    sk       = KernelMatrix(kernel, pts_sa, pts_sa)
    pre_time = @elapsed begin
      H  = assemble_hmatrix(sk; atol=solver.preconditioner.atol)
      Hf = try
        lu(H; atol=solver.preconditioner.luatol)
      catch
        @warn "Preconditioner factorization failed, falling back to identity matrix..."
        I
      end
    end
    verbose && @printf "Preconditioner assembly time: %1.3fs\n" pre_time
    (true, Hf)
  end

end

