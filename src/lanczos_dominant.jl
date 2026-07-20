
function _finalize_ritz_pairs(V, vals, vecs, k, iters)
  S_k = vecs[:, 1:k]
  RitzVecs = view(V, :, 1:iters) * S_k
  (vals[1:k], RitzVecs)
end

function partial_factorize(i::Int, dv, ev)
  T_k = SymTridiagonal(dv[1:i], ev[1:i-1])
  eig = eigen(T_k)
  p = sortperm(eig.values, rev=true)
  vals = eig.values[p]
  vecs = eig.vectors[:, p]
  (vals, vecs)
end

"""
`lanczos_dominant_eig(op, drop_tol::Float64, residual_tol::Float64; kwargs...)`

Computes a partial eigendecomposition of `op` using the Lanczos algorithm.
Dynamically identifies the rank `k` by finding eigenvalues `> drop_tol * λ_1`, 
then continues iterating until the residual bounds for all `k` eigenvalues 
are strictly less than `residual_tol`.
"""
function lanczos_dominant_eig(op, drop_tol::Float64, residual_tol::Float64=1e-13; 
                              v0=nothing, max_iter::Int=300, check_interval::Int=5)
  n = size(op, 1)
  v = v0 === nothing ? randn(n) : copy(v0)
  normalize!(v)
  T = eltype(v)
  V = zeros(T, n, max_iter)
  view(V,:,1) .= v
  w = zeros(T, n)
  dv = zeros(real(T), max_iter)   
  ev = zeros(real(T), max_iter)
  for i = 1:max_iter
    v_curr = view(V, :, i)
    mul!(w, op, v_curr)
    (i > 1) && axpy!(-ev[i-1], view(V, :, i-1), w)
    dv[i] = real(dot(v_curr, w))
    axpy!(-dv[i], v_curr, w)
    norm_before = norm(w)
    @inbounds for j = 1:i
      v_j = view(V, :, j)
      axpy!(-dot(v_j, w), v_j, w)
    end
    if norm(w) < 0.7 * norm_before
      @inbounds for j = 1:i
        v_j = view(V, :, j)
        axpy!(-dot(v_j, w), v_j, w)
      end
    end
    β_next = norm(w)
    ev[i]  = β_next 
    if β_next < eps(real(T))
      (vals, vecs) = partial_factorize(i, dv, ev)
      k = findlast(x -> x > drop_tol*vals[1], vals)
      k = isnothing(k) ? i : k
      return _finalize_ritz_pairs(V, vals, vecs, k, i)
    end
    if i < max_iter
      v_next = view(V, :, i+1)
      @inbounds for j in eachindex(v_next)
        v_next[j] = w[j]/β_next
      end
    end
    if i >= 2 && iszero(rem(i, check_interval))
      (vals, vecs) = partial_factorize(i, dv, ev)
      k = findlast(x -> x > drop_tol * vals[1], vals)
      if !isnothing(k)
        max_err = maximum(j->β_next*abs(vecs[end, j]), 1:k)
        max_err < residual_tol && return _finalize_ritz_pairs(V, vals, vecs, k, i)
      end
    end
  end
  @warn "Max iterations ($max_iter) reached before satisfying residual_tol ($residual_tol). Returning best available pairs."
  (vals, vecs) = partial_factorize(max_iter, dv, ev)
  k = findlast(x -> x > drop_tol * vals[1], vals)
  k = isnothing(k) ? max_iter : k
  _finalize_ritz_pairs(V, vals, vecs, k, max_iter)
end

