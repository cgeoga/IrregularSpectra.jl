
module IrregularSpectraNearestNeighborsExt

  using IrregularSpectra, NearestNeighbors
  using IrregularSpectra.Printf
  using IrregularSpectra.LinearAlgebra
  using IrregularSpectra.SparseArrays
  using NearestNeighbors.Distances
  using NearestNeighbors.StaticArrays
  import IrregularSpectra: SparsePreconditioner, GaussKernel, gen_kernel

  function IrregularSpectra.krylov_preconditioner!(pts_sa::Vector{SVector{D,Float64}}, 
                                                   Ω::NTuple{D,Float64}, 
                                                   solver::KrylovSolver{SparsePreconditioner, GaussKernel};
                                                   verbose=true) where{D}
    kernel = gen_kernel(solver, pts_sa, Ω)
    distf  = WeightedEuclidean((kernel.Mv.^2).*pi)
    k0     = kernel(pts_sa[1], pts_sa[1])
    pre_time = @elapsed begin
      tree   = KDTree(pts_sa, distf)
      ixs    = inrange(tree, pts_sa, sqrt(-log(solver.preconditioner.drop_tol*k0)))
      I      = reduce(vcat, ixs)
      J      = reduce(vcat, [fill(j, length(ixs[j])) for j in eachindex(ixs)])
      V      = [kernel(pts_sa[jk[1]], pts_sa[jk[2]]) for jk in zip(I,J)]
      M2ix   = sparse(I, J, V)
      Mf     = IrregularSpectra.LDivWrapper(ldlt(Symmetric(M2ix)))
    end
    verbose && @printf "preconditioner assembly time: %1.3fs\n" pre_time
    (true, Mf)
  end

end

