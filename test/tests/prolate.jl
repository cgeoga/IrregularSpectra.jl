
using StableRNGs, LinearAlgebra, IrregularSpectra

pts1 = sort(rand(StableRNG(123), 999)).*2.0 .- 8.0
pts2 = sort(rand(StableRNG(124), 800)).*2.0 .- 1.0
pts  = vcat(pts1, pts2)

win  = Prolate1D([(-8.0, -6.0), (-1.0, 1.0)])
Ω    = IrregularSpectra.default_Ω(pts, win)
wgr  = range(-Ω, Ω, length=2000)
rhs  = IrregularSpectra.linsys_rhs(win, wgr)

wts  = window_quadrature_weights(pts, win; verbose=false)[2]
@test maximum(norm, eachcol(wts)) < 0.2

out_ix = findall(w->abs(w) > 2*3.0, wgr)
F      = IrregularSpectra.nudftmatrix(wgr, pts, -1)
for wtsj in eachcol(wts)
  @test maximum(abs2, (F*wtsj)[out_ix]) < 1e-9
end

