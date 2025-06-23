
mean_and_std(v) = (mean(v), std(v))

function is_single_segment(seg1, seg2, candidate_dt, gaptol)
  (ds1, ds2) = (diff(seg1), diff(seg2))
  # check for the gap between seg1 and seg2:
  gapcheck  = abs(seg2[1] - seg1[end]) < gaptol*candidate_dt
  # check for max gaps:
  max1check = maximum(abs, ds1) <= candidate_dt*gaptol
  max2check = maximum(abs, ds2) <= candidate_dt*gaptol
  maxcheck  = max1check && max2check
  # check a simple z-test for different mean dt values:
  (mu1, sig1) = mean_and_std(ds1)
  (mu2, sig2) = mean_and_std(ds2)
  cmp         = (mu1 < mu2) ? Base.:< : Base.:>
  ztestcheck  = min(length(seg1), length(seg2)) <= 32 ? true : cmp(mu1 + 3*sig1, mu2 - 3*sig2)
  gapcheck && maxcheck && ztestcheck
end

function _interval_gaps(pts::Vector{Float64}, gaptol, minlen)

  issorted(pts) || throw(error("Please provide sorted points."))

  (length(pts) < minlen) && return [(0.0, 0.0)] # for pruning later.

  # Next, find the biggest gap and split into the two candidate segments.
  dpts = diff(pts)
  (biggest_gap, biggest_ix) = findmax(abs, dpts)
  seg1 = pts[1:biggest_ix]
  seg2 = pts[(biggest_ix+1):length(pts)]

  # Check if the extrema of the dt for either side of the biggest gap are within
  # tolerance for calling this a sigle segment. If so, we're done recursing.
  is_single_segment(seg1, seg2, median(dpts), gaptol) && return [(pts[1], pts[end])]

  # Otherwise, recursively apply this function to the two segments.
  iv1 = _interval_gaps(seg1, gaptol, minlen)
  iv2 = _interval_gaps(seg2, gaptol, minlen)
  reduce(vcat, (iv1, iv2))
end

function interval_coalesce!(intervals, tol)
  delete_ixs = Int64[]
  for j in 1:(length(intervals)-1)
    (iv1, iv2) = (intervals[j], intervals[j+1])
    if (iv2[1] - iv1[2]) < max(iv1[2]-iv1[1], iv2[2]-iv2[1])*tol
      intervals[j] = (iv1[1], iv2[2])
      push!(delete_ixs, j+1)
    end
  end
  deleteat!(intervals, delete_ixs)
  iszero(length(delete_ixs))
end

"""
`gappy_intervals(pts::Vector{Float64}; gaptol=8, minlen=128, coalesce_tol=0.005) -> Vector{NTuple{2,Float64}}`

Given a collection of (sorted!) 1D measurement locations `pts`, returns an attempt at automatically partitioning those points into sub-segments without large measurement gaps. 

Keyword arguments:

- `gaptol`: If the largest gap between adjacent points is bigger than `gaptol*median(abs, diff(pts))`, then split that domain to two intervals.

- `minlen`: If the resulting number of points in an interval after splitting is less than `minlen`, completely discard that interval.

- `coalesce_tol`: If, after partitioning, two intervals (a1, b1) and (a2, b2) are sufficiently close that (a2 - b1) < coalesce_tol*max(b1-a1, b2-a2)`, re-combine those intervals.

This is a **very heuristic** function, that is really offered just as a convenience for exploratory analysis. Please do not automatically use this on something important without inspecting the result.
"""
function gappy_intervals(pts::Vector{Float64}; gaptol=8, minlen=128, coalesce_tol=0.005)
  @warn "This functionality (`gappy_intervals)` is a work in progress! Please sanity check the result and use with caution. And open an issue if you hit an edge case that doesn't work well!" maxlog=1
  out = _interval_gaps(pts, gaptol, minlen)
  filter!(x->x!=(0.0, 0.0), out)
  reduced = interval_coalesce!(out, coalesce_tol)
  while !reduced
    reduced = interval_coalesce!(out, coalesce_tol)
  end
  out
end

