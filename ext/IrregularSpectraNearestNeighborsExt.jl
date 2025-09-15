
module IrregularSpectraNearestNeighborsExt

  using IrregularSpectra, NearestNeighbors
  using IrregularSpectra.Printf
  using IrregularSpectra.LinearAlgebra
  using IrregularSpectra.SparseArrays
  using NearestNeighbors.Distances
  using NearestNeighbors.StaticArrays
  import IrregularSpectra: SparsePreconditioner, SincKernel, GaussKernel, gen_kernel, kernel_tol_radius

  kerneldistance(kernel::GaussKernel) = WeightedEuclidean((kernel.Mv.^2).*pi)

  function IrregularSpectra.krylov_preconditioner!(pts_sa::Vector{SVector{D,Float64}}, 
                                                   Ω::NTuple{D,Float64}, 
                                                   solver::KrylovSolver{SparsePreconditioner, K};
                                                   verbose=true) where{D,K}
    kernel = gen_kernel(solver, pts_sa, Ω)
    distf  = kerneldistance(kernel)
    k0     = kernel(pts_sa[1], pts_sa[1])
    pre_time = @elapsed begin
      tree   = KDTree(pts_sa, distf)
      rad    = kernel_tol_radius(kernel, solver.preconditioner.drop_tol)
      ixs    = inrange(tree, pts_sa, rad)
      I      = reduce(vcat, ixs)
      J      = reduce(vcat, [fill(j, length(ixs[j])) for j in eachindex(ixs)])
      V      = [kernel(pts_sa[jk[1]], pts_sa[jk[2]]) for jk in zip(I,J)]
      M2ix   = sparse(I, J, V)
      Mf     = IrregularSpectra.LDivWrapper(cholesky(Symmetric(M2ix)))
    end
    verbose && @printf "preconditioner assembly time: %1.3fs\n" pre_time
    (true, Mf)
  end

end

