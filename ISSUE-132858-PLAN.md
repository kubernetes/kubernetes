# Issue #132858: Cache NodeInfo for Kubelet Admission

## Problem Statement

Currently in `pkg/kubelet/lifecycle/predicate.go:162-163`:
```go
pods := attrs.OtherPods
nodeInfo := schedulerframework.NewNodeInfo(pods...)
nodeInfo.SetNode(node)
```

`NewNodeInfo(pods...)` iterates through every resource in every container 3 times (desired, actual, allocated) for every pod. This is **O(n×m)** where n=pods, m=containers. It happens on:
- Every pod admission request
- Every pending resize retry

## Key Insight from PR Review

**SergeyKanzhelev's feedback is critical**: Sequential pod admissions will frequently invalidate the cache since each admission changes the pod set. The primary optimization benefit is for **resize retries**, where the same pod set is repeatedly evaluated while waiting for resources to become available.

## Recommended Approach

### 1. Cache Location

Add the cache to `predicateAdmitHandler` in `pkg/kubelet/lifecycle/predicate.go`:

```go
type predicateAdmitHandler struct {
    getNodeAnyWayFunc        getNodeAnyWayFuncType
    pluginResourceUpdateFunc pluginResourceUpdateFuncType
    admissionFailureHandler  AdmissionFailureHandler

    // NodeInfo cache
    mu              sync.RWMutex
    cachedNodeInfo  *schedulerframework.NodeInfo
    cacheGeneration uint64  // or use a hash of pod UIDs + resource versions
}
```

### 2. Cache Key/Invalidation Strategy

The cache should be invalidated when any of these change:

| Trigger | Detection Method |
|---------|------------------|
| Pod additions/removals | Hash of pod UIDs in `OtherPods` |
| Pod resource changes | Hash of pod ResourceVersions |
| Pod generation changes (resize) | Compare `pod.Generation` |
| Allocated resource changes | Hash of allocated resources |
| Node resource version | Compare `node.ResourceVersion` |

**Recommended hash approach**:
```go
type cacheKey struct {
    nodeResourceVersion string
    podSetHash          uint64  // FNV hash of sorted pod UIDs
    resourceHash        uint64  // FNV hash of pod resource versions
}
```

### 3. Implementation Steps

**Step 1**: Create cache infrastructure in `predicate.go`

```go
func (w *predicateAdmitHandler) getCachedNodeInfo(node *v1.Node, pods []*v1.Pod) *schedulerframework.NodeInfo {
    key := w.computeCacheKey(node, pods)

    w.mu.RLock()
    if w.cacheKey == key && w.cachedNodeInfo != nil {
        nodeInfoCacheHits.Inc()
        w.mu.RUnlock()
        return w.cachedNodeInfo
    }
    w.mu.RUnlock()

    // Cache miss - build new NodeInfo
    nodeInfoCacheMisses.Inc()
    nodeInfo := schedulerframework.NewNodeInfo(pods...)
    nodeInfo.SetNode(node)

    w.mu.Lock()
    w.cachedNodeInfo = nodeInfo
    w.cacheKey = key
    w.mu.Unlock()

    return nodeInfo
}
```

**Step 2**: Modify `Admit()` to use cache

```go
func (w *predicateAdmitHandler) Admit(attrs *PodAdmitAttributes) PodAdmitResult {
    // ... existing node fetch ...

    pods := attrs.OtherPods
    // CHANGE: Use cached NodeInfo
    nodeInfo := w.getCachedNodeInfo(node, pods)

    // ... rest of admission logic ...
}
```

**Step 3**: Handle cache invalidation after plugin resource update

The `pluginResourceUpdateFunc` modifies `nodeInfo.Allocatable`, so either:
- Clone the cached NodeInfo before passing to plugins, OR
- Invalidate cache after plugin updates

**Step 4**: Add metrics in `pkg/kubelet/metrics/metrics.go`

```go
var (
    nodeInfoCacheHits = prometheus.NewCounter(
        prometheus.CounterOpts{
            Name: "kubelet_admission_nodeinfo_cache_hits_total",
            Help: "Total number of NodeInfo cache hits during admission",
        },
    )
    nodeInfoCacheMisses = prometheus.NewCounter(
        prometheus.CounterOpts{
            Name: "kubelet_admission_nodeinfo_cache_misses_total",
            Help: "Total number of NodeInfo cache misses during admission",
        },
    )
)
```

### 4. Edge Cases to Handle

1. **`pluginResourceUpdateFunc` modifies NodeInfo**: The `sanitizeNodeAllocatable` in device manager modifies `nodeInfo.Allocatable`. Options:
   - Clone NodeInfo before plugin call (safest)
   - Track if plugin modified it and invalidate

2. **Synchronous node refetch on affinity failure** (lines 262-272): When admission fails due to node affinity, the code fetches the node synchronously and retries. The cache should handle this by:
   - Passing the new node to cache lookup
   - Cache miss due to different `node.ResourceVersion`

3. **Concurrent admission**: Multiple goroutines may call `Admit()` concurrently. Use proper locking (already shown above).

### 5. Testing Strategy

1. **Unit tests**: Mock `OtherPods` with varying pod sets, verify cache hits/misses
2. **Benchmark tests**: Measure admission latency with 100+ pods
3. **Integration tests**: Verify resize retries benefit from caching

### 6. Files to Modify

| File | Changes |
|------|---------|
| `pkg/kubelet/lifecycle/predicate.go` | Add cache logic, modify `Admit()` |
| `pkg/kubelet/metrics/metrics.go` | Add cache hit/miss counters |
| `pkg/kubelet/lifecycle/predicate_test.go` | Add cache tests |

## Alternative Approach: Cache at Allocation Manager Level

Since `canAdmitPod` and `canResizePod` both need NodeInfo, consider caching at `pkg/kubelet/allocation/allocation_manager.go` instead. This would:
- Centralize cache management
- Benefit both admission and resize flows
- Align with where `allocatedPods` is computed

## Expected Performance

Based on PR #134462 benchmarks:
- Cache hit: **O(1)** lookup vs **O(n×m)** iteration
- ~34.5 microseconds per admission with cache hits on 100-pod nodes

## Compatibility with Existing PR #134462

The existing PR takes a similar approach but has issues:
1. Contains merge commits (needs rebase)
2. Missing extended resource support in hash (per reviewer)
3. May be over-engineering invalidation triggers

Your implementation should:
- Start simpler with just pod UID + ResourceVersion hashing
- Add extended resource hashing per `bart0sh`'s feedback
- Ensure deterministic hashing with sorted keys

## Related Files

- `pkg/kubelet/lifecycle/predicate.go` - Main admission handler with NodeInfo construction
- `pkg/kubelet/allocation/allocation_manager.go` - Calls admission for pods and resizes
- `pkg/kubelet/kubelet_nodecache.go` - Existing node caching pattern to follow
- `pkg/scheduler/backend/cache/` - Scheduler cache pattern (more complex, for reference)

## References

- Issue: https://github.com/kubernetes/kubernetes/issues/132858
- Related PR: https://github.com/kubernetes/kubernetes/pull/134462
