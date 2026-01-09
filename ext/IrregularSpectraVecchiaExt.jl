
module IrregularSpectraVecchiaExt

  using IrregularSpectra, Vecchia
  using IrregularSpectra.LinearAlgebra
  using IrregularSpectra.SparseArrays
  using IrregularSpectra.Printf

  using IrregularSpectra: KrylovSolver, VecchiaPreconditioner, gen_kernel

  function IrregularSpectra.krylov_preconditioner!(pts_sa, Ω, solver::KrylovSolver{VecchiaPreconditioner,K}; 
                                                  verbose=false) where{K}
    kernel  = gen_kernel(solver, pts_sa, Ω)
    _kernel = (x, y, _) -> kernel(x, y) + Float64(x==y)*solver.perturbation
    pre_time = @elapsed begin
      cond   = KNNConditioning(solver.preconditioner.ncond)
      approx = VecchiaApproximation(pts_sa, _kernel; conditioning=cond, ordering=NoPermutation())
      pre    = rchol_preconditioner(approx, Float64[]).U
    end
    verbose && @printf "Preconditioner assembly time: %1.3fs\n" pre_time
    (false, pre)
  end

end

