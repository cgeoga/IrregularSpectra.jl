
module IrregularSpectraSparseExt

  using IrregularSpectra, NearestNeighbors
  using IrregularSpectra.Printf
  using IrregularSpectra.LinearAlgebra
  using IrregularSpectra.SparseArrays
  using NearestNeighbors.Distances
  import IrregularSpectra: SparsePreconditioner, gen_kernel

  # TODO (cg 2025/05/16 15:39): Weirdly, ldiv!(ch::CHOLMOD.Factor[...], rhs) or
  # whatever doesn't exist. So we need this little wrapper. I think this is
  # fixed in 1.12, but probably good to support LTS.
  struct LDivWrapper{T}
    M::T
  end

  Base.eltype(ldw::LDivWrapper{T}) where{T}  = eltype(ldw.M)
  Base.size(ldw::LDivWrapper{T}) where{T}    = size(ldw.M)
  Base.size(ldw::LDivWrapper{T}, j) where{T} = size(ldw.M, j)

  function LinearAlgebra.ldiv!(ldw::LDivWrapper{T}, v::Vector{ComplexF64}) where{T}
    tmp = ldw.M\v
    v  .= tmp
  end

  function LinearAlgebra.ldiv!(buf::Vector{ComplexF64}, ldw::LDivWrapper{T}, v::Vector{ComplexF64}) where{T}
    buf .= v
    ldiv!(ldw, buf)
  end

  function IrregularSpectra.krylov_preconditioner!(pts_sa, Ω, solver::KrylovSolver{SparsePreconditioner, K};
                                                   verbose=true) where{K}
    kernel = gen_kernel(solver, pts_sa, Ω)
    distf  = WeightedEuclidean((kernel.Mv.^2).*pi)
    pre_time = @elapsed begin
      tree   = KDTree(pts_sa, distf)
      ixs    = inrange(tree, pts_sa, sqrt(-log(solver.preconditioner.drop_tol)))
      I      = reduce(vcat, ixs)
      J      = reduce(vcat, [fill(j, length(ixs[j])) for j in eachindex(ixs)])
      V      = [kernel(pts_sa[jk[1]], pts_sa[jk[2]]) for jk in zip(I,J)]
      M2ix   = sparse(I, J, V)
      Mf     = LDivWrapper(ldlt(M2ix))
    end
    verbose && @printf "preconditioner assembly time: %1.3fs\n" pre_time
    (true, Mf)
  end

end

