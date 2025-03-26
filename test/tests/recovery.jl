
# A very simple test that the linear system to get the weights α has been solved
# to sufficient accuracy.

pts = sort(rand(StableRNG(123), 1000))
win = Kaiser(20.0)
Ω   = IrregularSpectra.default_Ω(pts, win)
wts = window_quadrature_weights(pts, win; verbose=false)[2]

wgrid  = range(-Ω, Ω, length=2*length(pts))
F      = IrregularSpectra.nudftmatrix(wgrid, pts, -1)
rec    = abs.(F*wts)
should = abs.(IrregularSpectra.fouriertransform(win, wgrid)) 

# test 1: in-bandwidth recovery, at least to single precision in inf norm:
ix_in = findall(x->abs(x) < IrregularSpectra.bandwidth(win), wgrid)
err   = abs.(rec[ix_in] - should[ix_in])./abs.(should[ix_in])
@test maximum(err) < 1e-8

# test 2: out-of-bandwidth control:
ix_out = findall(x->abs(x) > IrregularSpectra.bandwidth(win), wgrid)
@test maximum(rec[ix_out]) < 1.01*maximum(should[ix_out])

# test 3: Krylov variant. This one has to be a bit more slack, because asking an
# iterative solver to give you 1e-20 instead of 1e-16 is pretty hard.
wts_kry = window_quadrature_weights(pts, win; method=:krylov)[2]
rec_kry = abs2.(F*wts_kry)
@test maximum(rec_kry[ix_out]) < 1e-15

