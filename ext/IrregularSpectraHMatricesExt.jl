
module IrregularSpectraHMatricesExt

  using IrregularSpectra, HMatrices
  using IrregularSpectra.LinearAlgebra
  using IrregularSpectra.Printf

  using IrregularSpectra: KrylovSolver, HMatrixPreconditioner

  function IrregularSpectra.krylov_preconditioner!(pts_sa, Ω, solver::KrylovSolver{HMatrixPreconditioner,K}; 
                                                  verbose=false) where{K}
    if length(Ω) > 1 && K == IrregularSpectra.SincKernel
      @warn "In higher dimensions with the H-matrix preconditioner, please use the `GaussKernel` preconditioner kernel and not the `SincKernel` for better performance."
    end
    kernel   = IrregularSpectra.gen_kernel(solver, pts_sa, Ω)
    sk       = KernelMatrix(kernel, pts_sa, pts_sa)
    adm      = isone(length(Ω)) ? StrongAdmissibilityStd() : WeakAdmissibilityStd()
    @show adm
    pre_time = @elapsed begin
      H  = assemble_hmatrix(sk; rtol=solver.preconditioner.tol, atol=1e-8, adm=adm)
      Hf = try
        lu(H; rtol=solver.preconditioner.ftol, atol=1e-8)
      catch er
        @warn "Preconditioner factorization failed with error $er, falling back to identity matrix..."
        I
      end
    end
    verbose && @printf "Preconditioner assembly time: %1.3fs\n" pre_time
    (true, Hf)
  end

end

