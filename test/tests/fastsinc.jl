
offsets    = (0.0, -3.0, 25.0)
bandwidths = (0.1, 1.0, 10.0)

# centered on [-1/2, 1/2].
pts = sort(rand(StableRNG(123), 1000)) .- 0.5

# random input vector:
v   = randn(StableRNG(124), length(pts))

D   = Diagonal(rand(length(pts)))

for (oj, bk) in Iterators.product(offsets, bandwidths)
  ptsj = pts .+ oj
  fs   = IrregularSpectra.FastSinc(ptsj, bk, D=D)
  M    = D*[sinc(bk*(xj-xk)) for xj in ptsj, xk in ptsj]*D
  tmp  = similar(v)
  mul!(tmp, fs, v)
  @test maximum(abs, tmp-M*v) < 1e-12
end

