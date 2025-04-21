
module IrregularSpectraLowRankApproxExt

  using IrregularSpectra, LowRankApprox
  using IrregularSpectra.Printf

  using IrregularSpectra: gen_wgrid, linsys_rhs, NUFFT3, SketchSolver

  function IrregularSpectra.solve_linsys(pts, win, Ω, solver::SketchSolver; verbose=false)
      wgrid = gen_wgrid(pts, Ω) 
      b     = linsys_rhs(win, wgrid)
      F     = NUFFT3(pts, collect(wgrid.*(2*pi)), false, 1e-15)
      Fo    = LinearOperator(F)
      Fqr   = pqrfact(Fo; rtol=solver.tol)
      verbose && @printf "Rank of reduced QR: %i\n" size(Fqr.Q, 2)
      Fqr\b
  end

end

