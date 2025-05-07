
module IrregularSpectraVecchiaExt

  using IrregularSpectra, Vecchia
  using IrregularSpectra.LinearAlgebra
  using IrregularSpectra.SparseArrays
  using IrregularSpectra.Printf

  using IrregularSpectra: KrylovSolver, VecchiaPreconditioner, gen_kernel

  struct SparseInvCholesky
    Us::UpperTriangular{Float64, SparseMatrixCSC{Float64, Int64}}
  end

  LinearAlgebra.ishermitian(scp::SparseInvCholesky) = true
  LinearAlgebra.issymmetric(scp::SparseInvCholesky) = true
  Base.eltype(scp::SparseInvCholesky)  = Float64
  Base.size(scp::SparseInvCholesky)    = size(scp.Us)
  Base.size(scp::SparseInvCholesky, j) = size(scp.Us, j)
  function LinearAlgebra.mul!(buf, scp::SparseInvCholesky, x)
    tmp1 = scp.Us'*x
    mul!(buf, scp.Us, tmp1)
  end

  function IrregularSpectra.krylov_preconditioner!(pts_sa, Ω, solver::KrylovSolver{VecchiaPreconditioner,K}; 
                                                  verbose=false) where{K}
    kernel  = gen_kernel(solver, pts_sa, Ω)
    _kernel = (x, y, _) -> kernel(x, y) + Float64(x==y)*solver.perturbation
    multithread   = Threads.nthreads() > 1
    blas_nthreads = BLAS.get_num_threads()
    if multithread
      @info "Temporarily setting BLAS threads to 1 for parallel preconditioner assembly..." maxlog=1
      BLAS.set_num_threads(1)
    end
    pre_time = @elapsed begin
      cfg = Vecchia.fsa_knnconfig(Float64.(eachindex(pts_sa)), pts_sa, 
                                  solver.preconditioner.ncond, 
                                  solver.preconditioner.nfsa, 
                                  _kernel; randomize=false)
      p   = vec(Int64.(reduce(vcat, cfg.data)))
      permute!(pts_sa, p)
      Us  = sparse(Vecchia.rchol(cfg, Float64[]; use_tiles=true, issue_warning=false))
    end
    if multithread
      BLAS.set_num_threads(blas_nthreads)
    end
    verbose && @printf "Preconditioner assembly time: %1.3fs\n" pre_time
    (false, SparseInvCholesky(Us))
  end

end

