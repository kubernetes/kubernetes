# Fix for Issue #135487: resource.MustParse fails to parse quantities near math.MaxInt64

## Problem Statement
When parsing very large quantity values close to `math.MaxInt64`, the `resource.MustParse` function in `k8s.io/apimachinery/pkg/api/resource` fails to correctly interpret them.

## Root Cause
The issue is in the `ParseQuantity` function at line 302 in `quantity.go`. The problem occurs due to how `maxInt64Factors` (which is 18) and `precision` are calculated. For 19-digit integers, this leads to `infDecAmount` being used instead of `int64Amount`, causing the parsing failure.

## Solution
The fix adds a precise boundary check to ensure that quantities within the valid range of `[-math.MaxInt64, math.MaxInt64]` can be correctly parsed using the fast path (int64Amount), while values exceeding this range are still cleanly rejected.

### Fix Details
1. Added verification logic to check if a 19-digit number actually fits in int64
2. When maxInt64Factors yields -1, parse and verify if it fits in int64
3. If it fits, enable the fast path (precision = 0)
4. Keep minInt64 still in Dec path to avoid overflow

## Testing
- Values at math.MaxInt64 boundary
- Negative values at math.MinInt64 boundary
- Values just beyond boundaries
- Integration with CEL validation rules

## Related
- Kubernetes Issue: #135487
- Related PRs: #135602, #135634, #135595
