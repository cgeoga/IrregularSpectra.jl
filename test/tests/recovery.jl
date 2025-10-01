
# A very simple test that the linear system to get the weights α has been solved
# to sufficient accuracy.

for pre in (DensePreconditioner(), HMatrixPreconditioner(1e-10, 1e-10))

  pts   = sort(rand(StableRNG(123), 1000))
  win   = Kaiser(6.0)
  Ω     = IrregularSpectra.default_Ω(pts, win)
  wgrid = range(-Ω, Ω, length=2*length(pts))
  F     = IrregularSpectra.nudftmatrix(wgrid, pts, -1)

  wts = window_quadrature_weights(pts, win, solver=KrylovSolver(pre), Ω=Ω)[2][:,end]
  rec = abs.(F*wts)

  should = abs.(IrregularSpectra.fouriertransform.(Ref(win), wgrid)) 

  ix_in  = findall(x->abs(x) < IrregularSpectra.bandwidth(win), wgrid)
  ix_out = findall(x->abs(x) > IrregularSpectra.bandwidth(win), wgrid)

  # in-bandwidth recovery:
  err   = abs.(rec[ix_in] - should[ix_in])./abs.(should[ix_in])
  @test maximum(err) < 1e-3

  # out-of-bandwidth control:
  @test maximum(rec[ix_out])^2 < 1e-15
  @test maximum(rec[ix_out]) < 1.01*maximum(should[ix_out])

end

