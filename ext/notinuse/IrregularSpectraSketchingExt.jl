
module IrregularSpectraSketchingExt

  using IrregularSpectra, LowRankApprox#, BandlimitedOperators, WoodburyMatrices
  using IrregularSpectra.LinearAlgebra
  using IrregularSpectra.Printf

  using IrregularSpectra: gen_wgrid, glquadrule, fouriertransform, linsys_rhs, NUFFT3, IdentityKernel

  # TODO (cg 2025/06/25 14:03): fix this in BandlimitedOperators.jl
  function LinearAlgebra.ishermitian(fs::IrregularSpectra.BandlimitedOperators.NUFFT3)
    false
  end

  function IrregularSpectra.solve_linsys(pts, win, Ω, solver::SketchSolver{IdentityKernel}; 
                                         verbose=false)
    wgrid = gen_wgrid(pts, Ω) 
    b     = linsys_rhs(win, wgrid)
    F     = IrregularSpectra.BandlimitedOperators.NUFFT3(collect(wgrid.*(2*pi)), pts, -1)
    Fo    = LinearOperator(F)
    Fqr   = pqrfact(Fo; rtol=solver.sketchtol)
    verbose && @printf "Numerical rank: %i\n" size(Fqr.Q, 2)
    Fqr\b
  end

  # This method that specifically applies to the kernel-preconditioner approach
  # for KrylovSolvers seems to me to be of dubious value after some
  # experimentation. So I think I'll make the extension simpler for now and
  # withold it.
  #=
  function IrregularSpectra.solve_linsys(pts, win, Ω, solver::SketchSolver{K}; 
                                         verbose=false) where{K}
    (wgrid, glwts) = glquadrule(IrregularSpectra.krylov_nquad(pts, win), .-Ω, Ω)
    rhs       = linsys_rhs(win, wgrid)
    pts_sa    = IrregularSpectra.static_points(pts)
    kernel    = IrregularSpectra.gen_kernel(solver, pts_sa, Ω)
    Dv        = fouriertransform.(Ref(kernel), wgrid).*glwts
    op        = complex(Dv)
    ft1       = BandlimitedOperators.NUFFT3(wgrid.*(2*pi), pts_sa, false, 1e-15)
    fast      = FastBandlimited(true, (length(pts), length(pts)), ft1, ft1, wgrid, op)
    fasto     = HermitianLinearOperator(fast)
    (vals, U) = pheig(fasto; rtol=solver.sketchtol)
    wood      = Woodbury(Diagonal(fill(complex(solver.regularizer), length(pts))), 
                         U, Diagonal(complex(vals)), U')
    verbose && @printf "Numerical rank: %i\n" length(vals)
    rhs = vec(ft1'*(rhs.*Dv))
    wood\rhs
  end
  =#

end

