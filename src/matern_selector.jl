
function matern_cov(t, p)
  (sig, rho, v) = p
  iszero(t) && return sig^2
  arg = sqrt(2*v)*norm(t)/rho
  (sig^2)*((2^(1-v))/Bessels.gamma(v))*(arg^v)*Bessels.besselk(v, arg)
end

function matern_sdf(w, p) 
  d = length(w)
  (sig, rho, v) = p
  fpi2 = 4*pi^2
  pre  = (2^d)*(pi^(d/2))*Bessels.gamma(v+d/2)*((2*v)^v)/(Bessels.gamma(v)*rho^(2*v))
  (sig^2)*pre*(2*v/(rho^2) + fpi2*norm(w)^2)^(-v - d/2)
end

function matern_frequency_selector(pts, win, bw; a=minimum(pts), b=maximum(pts),
                                   nu_integerpart=0, alpha=0.9, alias_tol=1.0, 
                                   init_wmax=1.75*length(pts)/(pi*(b-a)))
  # this is basically exact even for non half-integer orders, but one can choose
  # a range parameter that breaks things, so for now we keep it restricted.
  in(nu_integerpart, (0,1,2)) || throw(error("For now, please provide nu_integerpart in (0,1,2)."))
  # depending on the order, we need a scale parameter to hopefully keep the
  # matrix from being too poorly conditioned.
  range_scale = (nu_integerpart == 0 ? 0.1 : (nu_integerpart == 1 ? 0.01 : 0.001))
  rho = (b - a)*range_scale
  # finally, we make the actual covariance function.
  kfn = (x,y) -> matern_cov(abs(x-y), (1.0, rho, nu_integerpart+1/2))
  # Next, we directly build the sparse Cholesky factor of the precision matrix,
  # getting a matrix such that Sigma ≈ inv(L'*inv(D)*L)).
  (L, D) = sparse_rchol(kfn, pts)
  # Now, we pick an ambitious wmax and check for aliasing control, and in the
  # case where we don't have control we shrink wmax by a factor of alpha.
  alias_error = Inf
  wmax        = init_wmax 
  while true
    (is_converged, weights) = opaqueweights(pts, win; wmax=wmax, verbose=false)
    estgrid = range(0.0, wmax/2, length=100)
    # get what the estimator should be, and check that everything is controlled
    # to tolerance alias_tol.
    wft = FourierTransform(win)
    nu  = nu_integerpart + 1/2
    shouldbe = map(estgrid) do vj
      quadgk(w->abs2(wft(w))*matern_sdf(w+vj, (1.0, rho, nu)), -bw, bw, atol=1e-12)[1]
    end
    # Now compute the full stochastic integrals by just evaluating the quadratic
    # form. Probably not fastest to form the full covariance matrix, but that is
    # an optimization we can do down the road.
    F  = nudftmatrix(estgrid, pts, 1)
    Mt = Diagonal(conj.(weights))*F'
    ests = diag(Mt'*(L\(D*(L'\Mt))))
    # Finally assess the error diagnostics.
    errors = abs.(ests .- shouldbe)
    all(errors .< shouldbe.*alias_tol) && break
    println("Reducing wmax ($wmax)...")
    wmax *= alpha
    wmax < 10.0 && throw(error("Could not achieve desired accuracy for any useful frequency limit."))
  end
  wmax 
end

# This is a very quick-and-dirty function to compute a sparse inverse Cholesky
# factor (or rather, an LDLt factorization with Sigma ≈ inv(L'*inv(D)*L)).  It
# is normally the backbone of Vecchia-type methods, but in this setting it is
# basically exact.
#
# TODO (cg 2024/11/21 11:41): this is my max-simple demonstration of the rchol.
# This could of course be optimized a great deal. Including:
# -- using banded matrices instead of sparse
# -- pre-allocating buffers and parallelizing 
# -- I'm sure other stuff too
# But I doubt that this will _ever_ be the bottleneck even as it is, so it
# doesn't seem like a high priority to me.
function sparse_rchol(kfn, pts)
  n = length(pts)
  L = sparse(Diagonal(ones(n)))
  D = zeros(n)
  for j in 1:n
    cix  = max(1, j-20):(j-1)
    pt   = pts[j]
    cpts = pts[cix]
    Sjj  = kfn(pt, pt)
    if j > 1
      Scc = cholesky!(Hermitian([kfn(x,y) for x in cpts, y in cpts]))
      Scj = [kfn(x, pt) for x in cpts]
      kweights = Scc\Scj
      ccov     = Sjj - dot(Scj, kweights)
      D[j]     = ccov
      view(L, j, cix) .= -kweights
    else
      D[j] = Sjj
    end
  end
  (LowerTriangular(L), Diagonal(D))
end

