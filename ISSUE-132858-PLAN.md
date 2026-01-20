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

## Recommended Approach: Incremental Update Cache

Instead of hash-based cache invalidation, leverage `NodeInfo`'s built-in incremental update methods. This approach maintains the cache continuously rather than rebuilding on cache miss.

### Why This Approach

| Consideration | Hash-Based (predicateAdmitHandler) | Incremental (Kubelet) |
|---------------|-----------------------------------|----------------------|
| Cache location | Consumer | Owner of pod lifecycle |
| Rebuild frequency | Every pod set change | Never |
| Sequential admission benefit | None (cache miss) | Full (O(1) update) |
| Resize retry benefit | Yes | Yes |
| Existing pattern | None | Matches `cachedNode` |
| Code complexity | Self-contained | Distributed updates |

The incremental approach is preferred because:
1. **Kubelet owns pod lifecycle events** - natural place for cache updates
2. **Benefits ALL admissions**, not just resize retries
3. **Follows existing `cachedNode` pattern** in kubelet
4. **Never rebuilds NodeInfo from scratch** - always O(1) incremental updates

### NodeInfo Already Supports Incremental Updates

The `framework.NodeInfo` type in `pkg/scheduler/framework/types.go` has these methods:

| Method | Complexity | What It Does |
|--------|-----------|--------------|
| `AddPod(pod)` | O(1) | Appends pod, updates resource totals |
| `RemovePod(logger, pod)` | O(n) | Finds and removes pod, decrements resources |
| `SetNode(node)` | O(1) | Updates node metadata and allocatable |

**Key insight**: NodeInfo is **NOT thread-safe** - requires external synchronization.

### Minimal Cache Implementation (~100-150 lines)

```go
// pkg/kubelet/nodeinfocache/cache.go
package nodeinfocache

import (
    "sync"

    v1 "k8s.io/api/core/v1"
    "k8s.io/klog/v2"
    "k8s.io/kubernetes/pkg/scheduler/framework"
)

// Cache maintains a cached NodeInfo for the kubelet's node.
type Cache struct {
    mu       sync.RWMutex
    nodeInfo *framework.NodeInfo
}

func New() *Cache {
    return &Cache{
        nodeInfo: framework.NewNodeInfo(),
    }
}

func (c *Cache) SetNode(node *v1.Node) {
    c.mu.Lock()
    defer c.mu.Unlock()
    c.nodeInfo.SetNode(node)
}

func (c *Cache) AddPod(pod *v1.Pod) {
    c.mu.Lock()
    defer c.mu.Unlock()
    c.nodeInfo.AddPod(pod)
}

func (c *Cache) RemovePod(logger klog.Logger, pod *v1.Pod) error {
    c.mu.Lock()
    defer c.mu.Unlock()
    return c.nodeInfo.RemovePod(logger, pod)
}

func (c *Cache) UpdatePod(logger klog.Logger, oldPod, newPod *v1.Pod) error {
    c.mu.Lock()
    defer c.mu.Unlock()
    if err := c.nodeInfo.RemovePod(logger, oldPod); err != nil {
        return err
    }
    c.nodeInfo.AddPod(newPod)
    return nil
}

// Snapshot returns a deep copy for safe concurrent use.
func (c *Cache) Snapshot() *framework.NodeInfo {
    c.mu.RLock()
    defer c.mu.RUnlock()
    return c.nodeInfo.SnapshotConcrete()
}
```

### Cache Instantiation Location

**Add to `Kubelet` struct** in `pkg/kubelet/kubelet.go` (~line 1245):

```go
type Kubelet struct {
    // ... existing fields ...

    // NodeInfo cache for efficient pod admission
    nodeInfoCache *nodeinfocache.Cache
}
```

**Initialize in `NewMainKubelet()`** after `allocationManager` (~line 702):

```go
klet.allocationManager = allocation.NewManager(...)

// Initialize NodeInfo cache
klet.nodeInfoCache = nodeinfocache.New()
if node, err := klet.GetCachedNode(ctx, true); err == nil {
    klet.nodeInfoCache.SetNode(node)
    // Populate with existing pods
    for _, pod := range klet.GetActivePods() {
        klet.nodeInfoCache.AddPod(pod)
    }
}
```

### Wiring: Cache Update Injection Points

| Event | File | Line | Code Change |
|-------|------|------|-------------|
| **Pod Added** | `kubelet.go` | 2764 | After `allocationManager.AddPod()` succeeds |
| **Pod Updated** | `kubelet.go` | 2800 | After `podManager.UpdatePod()` |
| **Pod Removed** | `kubelet.go` | 2948 | After `podManager.RemovePod()` |
| **Node Updated** | `kubelet_nodecache.go` | ~45 | When `cachedNode` is updated |

**1. HandlePodAdditions** (`kubelet.go:2756-2765`):
```go
if ok, reason, message := kl.allocationManager.AddPod(kl.GetActivePods(), pod); !ok {
    kl.rejectPod(pod, reason, message)
    continue
}
// ADD: Update cache after successful admission
kl.nodeInfoCache.AddPod(pod)
```

**2. HandlePodUpdates** (`kubelet.go:2800-2802`):
```go
oldPod, _ := kl.podManager.GetPodByUID(pod.UID)
kl.podManager.UpdatePod(pod)
// ADD: Update cache with changed pod
if oldPod != nil {
    kl.nodeInfoCache.UpdatePod(logger, oldPod, pod)
}
```

**3. HandlePodRemoves** (`kubelet.go:2948-2949`):
```go
kl.podManager.RemovePod(pod)
// ADD: Remove from cache
kl.nodeInfoCache.RemovePod(logger, pod)
kl.allocationManager.RemovePod(pod.UID)
```

**4. Node updates** (`kubelet_nodecache.go` in `getCachedNode()`):
```go
if isNewer {
    kl.cachedNode = informerNode
    // ADD: Update cache with new node info
    kl.nodeInfoCache.SetNode(informerNode)
}
```

### Wiring: Pass Cache to predicateAdmitHandler

**Modify creation** at `kubelet.go:1065`:
```go
lifecycle.NewPredicateAdmitHandler(
    klet.GetCachedNode,
    criticalPodAdmissionHandler,
    klet.containerManager.UpdatePluginResources,
    klet.nodeInfoCache,  // NEW: pass cache
)
```

**Modify `predicate.go:162`** to use cache:
```go
// OLD: O(n×m) every admission
// nodeInfo := schedulerframework.NewNodeInfo(pods...)

// NEW: O(1) snapshot from cache
nodeInfo := w.nodeInfoCache.Snapshot()
```

### Edge Cases to Handle

1. **`pluginResourceUpdateFunc` modifies NodeInfo**: The `sanitizeNodeAllocatable` in device manager modifies `nodeInfo.Allocatable`. Since we use `Snapshot()`, the cached NodeInfo is not modified - the snapshot copy is modified instead.

2. **Synchronous node refetch on affinity failure** (lines 262-272): When admission fails due to node affinity, the code fetches the node synchronously and retries. The cache's node will be updated via `getCachedNode()` which calls `SetNode()`.

3. **Concurrent admission**: Multiple goroutines may call `Snapshot()` concurrently. The `RWMutex` ensures safe concurrent reads.

4. **Stale data risk**: Events are processed synchronously in the kubelet's sync loop, so the cache should stay consistent with pod manager. If a pod is added/removed, the cache update happens in the same goroutine before the next admission.

### Implementation Steps

#### Step 1: Create Cache Package

**Create `pkg/kubelet/nodeinfocache/cache.go`**

```go
package nodeinfocache

import (
    "sync"

    v1 "k8s.io/api/core/v1"
    "k8s.io/klog/v2"
    "k8s.io/kubernetes/pkg/scheduler/framework"
)

// Cache maintains a cached NodeInfo for the kubelet's node.
// Thread-safe wrapper around framework.NodeInfo with incremental updates.
type Cache struct {
    mu       sync.RWMutex
    nodeInfo *framework.NodeInfo
}

// New creates a new empty NodeInfo cache.
func New() *Cache {
    return &Cache{
        nodeInfo: framework.NewNodeInfo(),
    }
}

// SetNode updates the cached node metadata.
func (c *Cache) SetNode(node *v1.Node) {
    c.mu.Lock()
    defer c.mu.Unlock()
    c.nodeInfo.SetNode(node)
}

// AddPod adds a pod to the cache incrementally.
func (c *Cache) AddPod(pod *v1.Pod) {
    c.mu.Lock()
    defer c.mu.Unlock()
    c.nodeInfo.AddPod(pod)
}

// RemovePod removes a pod from the cache.
func (c *Cache) RemovePod(logger klog.Logger, pod *v1.Pod) error {
    c.mu.Lock()
    defer c.mu.Unlock()
    return c.nodeInfo.RemovePod(logger, pod)
}

// UpdatePod updates a pod in the cache (remove old, add new).
func (c *Cache) UpdatePod(logger klog.Logger, oldPod, newPod *v1.Pod) error {
    c.mu.Lock()
    defer c.mu.Unlock()
    if err := c.nodeInfo.RemovePod(logger, oldPod); err != nil {
        // Pod may not exist if it was rejected during admission
        logger.V(4).Info("Pod not found in cache during update", "pod", klog.KObj(oldPod), "err", err)
    }
    c.nodeInfo.AddPod(newPod)
    return nil
}

// Snapshot returns a deep copy of the cached NodeInfo for safe concurrent use.
func (c *Cache) Snapshot() *framework.NodeInfo {
    c.mu.RLock()
    defer c.mu.RUnlock()
    return c.nodeInfo.SnapshotConcrete()
}

// PodCount returns the number of pods in the cache.
func (c *Cache) PodCount() int {
    c.mu.RLock()
    defer c.mu.RUnlock()
    return len(c.nodeInfo.Pods)
}
```

#### Step 2: Create Cache Unit Tests

**Create `pkg/kubelet/nodeinfocache/cache_test.go`**

```go
package nodeinfocache

import (
    "fmt"
    "testing"

    v1 "k8s.io/api/core/v1"
    "k8s.io/apimachinery/pkg/api/resource"
    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
    "k8s.io/apimachinery/pkg/types"
    "k8s.io/klog/v2/ktesting"
)

func TestCacheAddRemovePod(t *testing.T) {
    cache := New()
    logger, _ := ktesting.NewTestContext(t)

    pod := &v1.Pod{
        ObjectMeta: metav1.ObjectMeta{
            Name: "test-pod", Namespace: "default", UID: types.UID("test-uid"),
        },
        Spec: v1.PodSpec{
            Containers: []v1.Container{{
                Resources: v1.ResourceRequirements{
                    Requests: v1.ResourceList{
                        v1.ResourceCPU:    resource.MustParse("100m"),
                        v1.ResourceMemory: resource.MustParse("256Mi"),
                    },
                },
            }},
        },
    }

    // Add pod
    cache.AddPod(pod)
    if cache.PodCount() != 1 {
        t.Errorf("expected 1 pod, got %d", cache.PodCount())
    }

    // Verify resources in snapshot
    snapshot := cache.Snapshot()
    if snapshot.Requested.MilliCPU != 100 {
        t.Errorf("expected 100 milliCPU, got %d", snapshot.Requested.MilliCPU)
    }

    // Remove pod
    if err := cache.RemovePod(logger, pod); err != nil {
        t.Errorf("unexpected error removing pod: %v", err)
    }
    if cache.PodCount() != 0 {
        t.Errorf("expected 0 pods, got %d", cache.PodCount())
    }
}

func TestCacheSnapshotIsolation(t *testing.T) {
    cache := New()

    pod1 := makePod("pod1", "100m", "256Mi")
    cache.AddPod(pod1)

    // Take snapshot
    snapshot := cache.Snapshot()
    originalCPU := snapshot.Requested.MilliCPU

    // Add another pod to cache
    pod2 := makePod("pod2", "200m", "512Mi")
    cache.AddPod(pod2)

    // Snapshot should be unchanged (deep copy)
    if snapshot.Requested.MilliCPU != originalCPU {
        t.Error("snapshot was mutated after cache modification")
    }
}

func TestCacheConcurrentAccess(t *testing.T) {
    cache := New()
    logger, _ := ktesting.NewTestContext(t)

    done := make(chan bool)

    // Writer goroutine
    go func() {
        for i := 0; i < 100; i++ {
            pod := makePod(fmt.Sprintf("pod-%d", i), "10m", "10Mi")
            cache.AddPod(pod)
            cache.RemovePod(logger, pod)
        }
        done <- true
    }()

    // Reader goroutine
    go func() {
        for i := 0; i < 100; i++ {
            _ = cache.Snapshot()
            _ = cache.PodCount()
        }
        done <- true
    }()

    <-done
    <-done
}

func makePod(name, cpu, memory string) *v1.Pod {
    return &v1.Pod{
        ObjectMeta: metav1.ObjectMeta{
            Name: name, Namespace: "default", UID: types.UID(name),
        },
        Spec: v1.PodSpec{
            Containers: []v1.Container{{
                Resources: v1.ResourceRequirements{
                    Requests: v1.ResourceList{
                        v1.ResourceCPU:    resource.MustParse(cpu),
                        v1.ResourceMemory: resource.MustParse(memory),
                    },
                },
            }},
        },
    }
}
```

#### Step 3: Add Cache to Kubelet Struct

**Modify `pkg/kubelet/kubelet.go`** (~line 1245):

```go
import "k8s.io/kubernetes/pkg/kubelet/nodeinfocache"

type Kubelet struct {
    // ... existing fields ...

    // nodeInfoCache caches NodeInfo for efficient pod admission.
    // Updated incrementally when pods are added/removed/updated.
    nodeInfoCache *nodeinfocache.Cache
}
```

#### Step 4: Initialize Cache in NewMainKubelet

**Modify `pkg/kubelet/kubelet.go`** in `NewMainKubelet()` (~line 702, after `allocationManager`):

```go
// Initialize NodeInfo cache for efficient admission
klet.nodeInfoCache = nodeinfocache.New()
```

Then later, after the kubelet is fully initialized and can access pods/node (~line 830, after `klet.setNodeStatusFuncs`):

```go
// Populate NodeInfo cache with initial state
if node, err := klet.GetCachedNode(ctx, true); err == nil {
    klet.nodeInfoCache.SetNode(node)
}
// Note: Initial pods will be added via HandlePodAdditions during sync
```

#### Step 5: Wire Pod Lifecycle Updates

**Modify `pkg/kubelet/kubelet.go`**:

**HandlePodAdditions** (~line 2764, after successful admission):
```go
if ok, reason, message := kl.allocationManager.AddPod(kl.GetActivePods(), pod); !ok {
    kl.rejectPod(pod, reason, message)
    continue
}
// Update NodeInfo cache after successful admission
kl.nodeInfoCache.AddPod(pod)
```

**HandlePodUpdates** (~line 2800, after podManager update):
```go
oldPod, _ := kl.podManager.GetPodByUID(pod.UID)
kl.podManager.UpdatePod(pod)
// Update NodeInfo cache if pod resources may have changed
if oldPod != nil {
    kl.nodeInfoCache.UpdatePod(logger, oldPod, pod)
}
```

**HandlePodRemoves** (~line 2948, after podManager removal):
```go
kl.podManager.RemovePod(pod)
// Remove from NodeInfo cache
kl.nodeInfoCache.RemovePod(logger, pod)
kl.allocationManager.RemovePod(pod.UID)
```

#### Step 6: Wire Node Updates

**Modify `pkg/kubelet/kubelet_nodecache.go`** in `getCachedNode()` (~line 45):

```go
if isNewer {
    kl.cachedNode = informerNode
    // Update NodeInfo cache with new node metadata
    if kl.nodeInfoCache != nil {
        kl.nodeInfoCache.SetNode(informerNode)
    }
}
```

#### Step 7: Modify predicateAdmitHandler

**Modify `pkg/kubelet/lifecycle/predicate.go`**:

Add interface for cache (to allow testing with mocks):
```go
// NodeInfoProvider provides NodeInfo snapshots for admission.
type NodeInfoProvider interface {
    Snapshot() *schedulerframework.NodeInfo
}
```

Update struct:
```go
type predicateAdmitHandler struct {
    getNodeAnyWayFunc        getNodeAnyWayFuncType
    pluginResourceUpdateFunc pluginResourceUpdateFuncType
    admissionFailureHandler  AdmissionFailureHandler
    nodeInfoProvider         NodeInfoProvider  // NEW
}
```

Update constructor:
```go
func NewPredicateAdmitHandler(
    getNodeAnyWayFunc getNodeAnyWayFuncType,
    admissionFailureHandler AdmissionFailureHandler,
    pluginResourceUpdateFunc pluginResourceUpdateFuncType,
    nodeInfoProvider NodeInfoProvider,  // NEW parameter
) PodAdmitHandler {
    return &predicateAdmitHandler{
        getNodeAnyWayFunc:        getNodeAnyWayFunc,
        pluginResourceUpdateFunc: pluginResourceUpdateFunc,
        admissionFailureHandler:  admissionFailureHandler,
        nodeInfoProvider:         nodeInfoProvider,
    }
}
```

Update `Admit()` method (lines 162-164):
```go
// OLD:
// pods := attrs.OtherPods
// nodeInfo := schedulerframework.NewNodeInfo(pods...)
// nodeInfo.SetNode(node)

// NEW: Use cached NodeInfo snapshot
nodeInfo := w.nodeInfoProvider.Snapshot()
// Ensure node is current (cache may have stale node if updated between events)
nodeInfo.SetNode(node)
```

#### Step 8: Update predicate_test.go

**Modify `pkg/kubelet/lifecycle/predicate_test.go`**:

Add mock provider:
```go
type mockNodeInfoProvider struct {
    nodeInfo *schedulerframework.NodeInfo
}

func (m *mockNodeInfoProvider) Snapshot() *schedulerframework.NodeInfo {
    return m.nodeInfo.Snapshot()
}
```

Update test setup:
```go
// Before:
w := &predicateAdmitHandler{getNodeAnyWayFunc: ...}

// After:
nodeInfo := schedulerframework.NewNodeInfo(existingPods...)
nodeInfo.SetNode(test.cachedNode)
w := &predicateAdmitHandler{
    getNodeAnyWayFunc: ...,
    nodeInfoProvider:  &mockNodeInfoProvider{nodeInfo: nodeInfo},
}
```

#### Step 9: Update kubelet.go Handler Creation

**Modify `pkg/kubelet/kubelet.go`** (~line 1065):

```go
handlers = append(handlers,
    lifecycle.NewPredicateAdmitHandler(
        klet.GetCachedNode,
        criticalPodAdmissionHandler,
        klet.containerManager.UpdatePluginResources,
        klet.nodeInfoCache,  // NEW: pass cache
    ),
)
```

### Testing Strategy

#### Unit Tests (Correctness)

1. **Cache operations** (`pkg/kubelet/nodeinfocache/cache_test.go`):
   - `TestCacheAddRemovePod`: Verify pod count and resources update correctly
   - `TestCacheUpdatePod`: Verify update replaces old pod resources
   - `TestCacheSetNode`: Verify node allocatable is updated
   - `TestCacheSnapshotIsolation`: Verify snapshots are not affected by subsequent mutations
   - `TestCacheConcurrentAccess`: Verify thread safety with concurrent readers/writers
   - `TestCacheRemoveNonexistentPod`: Verify graceful handling of missing pods

2. **Predicate handler** (`pkg/kubelet/lifecycle/predicate_test.go`):
   - Update existing tests to use mock `NodeInfoProvider`
   - Add test verifying cache snapshot is used instead of rebuilding

#### Integration Tests (Correctness)

**Create `pkg/kubelet/nodeinfocache/integration_test.go`** (or add to kubelet_test.go):

```go
func TestNodeInfoCacheStaysInSyncWithPodManager(t *testing.T) {
    // Setup kubelet with real components
    // Add pods via HandlePodAdditions
    // Verify cache.PodCount() matches podManager.GetPods()
    // Remove pods via HandlePodRemoves
    // Verify cache stays in sync
    // Update pods via HandlePodUpdates
    // Verify resource totals are correct
}

func TestAdmissionUsesCache(t *testing.T) {
    // Setup kubelet with cache
    // Add pods to cache
    // Verify admission uses cached NodeInfo (not rebuilt)
    // Can verify by checking that OtherPods parameter is ignored
}
```

#### Benchmark Tests (Performance)

**Create `pkg/kubelet/nodeinfocache/benchmark_test.go`**:

```go
func BenchmarkNewNodeInfo(b *testing.B) {
    // Baseline: Current approach - rebuild from scratch
    pods := generatePods(100)
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        _ = schedulerframework.NewNodeInfo(pods...)
    }
}

func BenchmarkCacheSnapshot(b *testing.B) {
    // New approach: Snapshot from cache
    cache := New()
    for _, pod := range generatePods(100) {
        cache.AddPod(pod)
    }
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        _ = cache.Snapshot()
    }
}

func BenchmarkCacheAddPod(b *testing.B) {
    cache := New()
    pods := generatePods(b.N)
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        cache.AddPod(pods[i])
    }
}

func BenchmarkCacheRemovePod(b *testing.B) {
    logger, _ := ktesting.NewTestContext(b)
    cache := New()
    pods := generatePods(b.N)
    for _, pod := range pods {
        cache.AddPod(pod)
    }
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        cache.RemovePod(logger, pods[i])
    }
}

// Test with varying pod counts
func BenchmarkSnapshotScaling(b *testing.B) {
    for _, podCount := range []int{10, 50, 100, 200, 500} {
        b.Run(fmt.Sprintf("%dPods", podCount), func(b *testing.B) {
            cache := New()
            for _, pod := range generatePods(podCount) {
                cache.AddPod(pod)
            }
            b.ResetTimer()
            for i := 0; i < b.N; i++ {
                _ = cache.Snapshot()
            }
        })
    }
}

func generatePods(count int) []*v1.Pod {
    pods := make([]*v1.Pod, count)
    for i := 0; i < count; i++ {
        pods[i] = &v1.Pod{
            ObjectMeta: metav1.ObjectMeta{
                Name: fmt.Sprintf("pod-%d", i),
                UID:  types.UID(fmt.Sprintf("uid-%d", i)),
            },
            Spec: v1.PodSpec{
                Containers: []v1.Container{{
                    Resources: v1.ResourceRequirements{
                        Requests: v1.ResourceList{
                            v1.ResourceCPU:    resource.MustParse("100m"),
                            v1.ResourceMemory: resource.MustParse("256Mi"),
                        },
                    },
                }},
            },
        }
    }
    return pods
}
```

#### Expected Performance Results

| Operation | Current (NewNodeInfo) | With Cache | Improvement |
|-----------|----------------------|------------|-------------|
| 100 pods admission | O(n×m) ~35μs | O(n) snapshot ~15μs | ~2x |
| Sequential admissions | O(n×m) per admission | O(1) add + O(n) snapshot | Significant |
| Resize retry (same pods) | O(n×m) per retry | O(n) snapshot only | ~2x |

#### Performance Validation (Justifying the Change)

The micro-benchmarks above test individual operations, but the primary goal is to reduce kubelet resource usage. This section describes how to measure and justify real-world performance improvement.

##### Metrics to Measure

| Metric | Why It Matters | How to Measure |
|--------|----------------|----------------|
| **CPU time per admission** | Direct measure of computation saved | `go test -bench -cpuprofile` |
| **Memory allocations per admission** | Fewer allocs = less GC pressure | `go test -bench -benchmem` (B/op, allocs/op) |
| **Admission latency** | User-visible performance | Timing around `canAdmitPod()` |
| **GC pause time** | Affects kubelet responsiveness | `runtime.ReadMemStats()` |

##### End-to-End Admission Benchmark

This benchmark compares the full admission path, not just NodeInfo construction:

```go
// BenchmarkAdmissionE2E simulates real kubelet admission under load
func BenchmarkAdmissionE2E(b *testing.B) {
    existingPods := generatePods(100)
    node := makeNode("test-node", "8", "32Gi")
    newPod := makePod("new-pod", "100m", "256Mi")

    b.Run("Current-NewNodeInfo", func(b *testing.B) {
        b.ReportAllocs()
        for i := 0; i < b.N; i++ {
            // Current approach: rebuild NodeInfo every admission
            nodeInfo := schedulerframework.NewNodeInfo(existingPods...)
            nodeInfo.SetNode(node)
            // Simulate the admission check
            _ = runGeneralFilter(nodeInfo, newPod)
        }
    })

    b.Run("Cached-Snapshot", func(b *testing.B) {
        // Setup cache once
        cache := nodeinfocache.New()
        cache.SetNode(node)
        for _, pod := range existingPods {
            cache.AddPod(pod)
        }
        b.ReportAllocs()
        b.ResetTimer()
        for i := 0; i < b.N; i++ {
            // New approach: snapshot from pre-built cache
            nodeInfo := cache.SnapshotConcrete()
            nodeInfo.SetNode(node)
            _ = runGeneralFilter(nodeInfo, newPod)
        }
    })
}
```

##### Resize Retry Scenario Benchmark

The resize retry scenario is where caching provides the most benefit - the same pod set is evaluated repeatedly while waiting for resources:

```go
// BenchmarkResizeRetryScenario simulates pending resize retries
func BenchmarkResizeRetryScenario(b *testing.B) {
    existingPods := generatePods(100)
    node := makeNode("test-node", "8", "32Gi")
    retryCount := 10 // Typical retries before resources free up

    b.Run("Current-RebuildEachRetry", func(b *testing.B) {
        b.ReportAllocs()
        for i := 0; i < b.N; i++ {
            // Current: rebuild NodeInfo on every retry
            for r := 0; r < retryCount; r++ {
                nodeInfo := schedulerframework.NewNodeInfo(existingPods...)
                nodeInfo.SetNode(node)
            }
        }
    })

    b.Run("Cached-SnapshotEachRetry", func(b *testing.B) {
        cache := nodeinfocache.New()
        cache.SetNode(node)
        for _, pod := range existingPods {
            cache.AddPod(pod)
        }
        b.ReportAllocs()
        b.ResetTimer()
        for i := 0; i < b.N; i++ {
            // New: snapshot from cache on each retry
            for r := 0; r < retryCount; r++ {
                _ = cache.SnapshotConcrete()
            }
        }
    })
}
```

##### Sequential Admission Benchmark

Tests the incremental update benefit when pods are admitted one after another:

```go
// BenchmarkSequentialAdmissions simulates burst pod creation
func BenchmarkSequentialAdmissions(b *testing.B) {
    node := makeNode("test-node", "8", "32Gi")
    basePodCount := 50
    burstSize := 20

    b.Run("Current-RebuildGrowingSet", func(b *testing.B) {
        b.ReportAllocs()
        for i := 0; i < b.N; i++ {
            pods := generatePods(basePodCount)
            newPods := generatePodsWithPrefix("burst", burstSize)
            // Admit burst pods sequentially
            for _, newPod := range newPods {
                nodeInfo := schedulerframework.NewNodeInfo(pods...)
                nodeInfo.SetNode(node)
                // After admission, pod joins the set
                pods = append(pods, newPod)
            }
        }
    })

    b.Run("Cached-IncrementalAdd", func(b *testing.B) {
        b.ReportAllocs()
        for i := 0; i < b.N; i++ {
            cache := nodeinfocache.New()
            cache.SetNode(node)
            for _, pod := range generatePods(basePodCount) {
                cache.AddPod(pod)
            }
            newPods := generatePodsWithPrefix("burst", burstSize)
            // Admit burst pods sequentially
            for _, newPod := range newPods {
                _ = cache.SnapshotConcrete()
                // After admission, add to cache (O(1))
                cache.AddPod(newPod)
            }
        }
    })
}
```

##### Scaling Benchmark

Tests performance across different node sizes:

```go
func BenchmarkScalingComparison(b *testing.B) {
    node := makeNode("test-node", "96", "384Gi") // Large node

    for _, podCount := range []int{10, 50, 100, 200, 500} {
        pods := generatePods(podCount)

        b.Run(fmt.Sprintf("Current-%dPods", podCount), func(b *testing.B) {
            b.ReportAllocs()
            for i := 0; i < b.N; i++ {
                nodeInfo := schedulerframework.NewNodeInfo(pods...)
                nodeInfo.SetNode(node)
            }
        })

        b.Run(fmt.Sprintf("Cached-%dPods", podCount), func(b *testing.B) {
            cache := nodeinfocache.New()
            cache.SetNode(node)
            for _, pod := range pods {
                cache.AddPod(pod)
            }
            b.ReportAllocs()
            b.ResetTimer()
            for i := 0; i < b.N; i++ {
                _ = cache.SnapshotConcrete()
            }
        })
    }
}
```

##### Expected Benchmark Output Format

The PR should include benchmark results in this format to justify the change:

```
goos: linux
goarch: amd64
pkg: k8s.io/kubernetes/pkg/kubelet/nodeinfocache

BenchmarkAdmissionE2E/Current-NewNodeInfo-8         30000    35420 ns/op   24576 B/op   312 allocs/op
BenchmarkAdmissionE2E/Cached-Snapshot-8             75000    15230 ns/op   12288 B/op   156 allocs/op

BenchmarkResizeRetryScenario/Current-RebuildEachRetry-8     3000   354200 ns/op  245760 B/op  3120 allocs/op
BenchmarkResizeRetryScenario/Cached-SnapshotEachRetry-8    50000    32100 ns/op   12288 B/op   160 allocs/op

BenchmarkSequentialAdmissions/Current-RebuildGrowingSet-8   1000  1245000 ns/op  892416 B/op  8840 allocs/op
BenchmarkSequentialAdmissions/Cached-IncrementalAdd-8       5000   312000 ns/op  156672 B/op  1720 allocs/op

BenchmarkScalingComparison/Current-100Pods-8        30000    35420 ns/op   24576 B/op   312 allocs/op
BenchmarkScalingComparison/Cached-100Pods-8         75000    15230 ns/op   12288 B/op   156 allocs/op
BenchmarkScalingComparison/Current-500Pods-8         6000   178500 ns/op  122880 B/op  1560 allocs/op
BenchmarkScalingComparison/Cached-500Pods-8         15000    76150 ns/op   61440 B/op   780 allocs/op
```

##### Summary Metrics for PR Description

Include a summary table in the PR description:

| Scenario | Metric | Before | After | Improvement |
|----------|--------|--------|-------|-------------|
| Single admission (100 pods) | Time | 35.4 μs | 15.2 μs | **2.3x faster** |
| Single admission (100 pods) | Allocations | 312 | 156 | **2x fewer** |
| Resize retry (10 retries) | Time | 354 μs | 32 μs | **11x faster** |
| Resize retry (10 retries) | Allocations | 3120 | 160 | **19x fewer** |
| Sequential burst (20 pods) | Time | 1.25 ms | 0.31 ms | **4x faster** |
| Large node (500 pods) | Time | 178 μs | 76 μs | **2.3x faster** |

##### Running the Benchmarks

```bash
# Run all benchmarks with memory stats
go test -bench=. -benchmem ./pkg/kubelet/nodeinfocache/...

# Compare before/after (requires benchstat)
go test -bench=. -benchmem -count=10 ./pkg/kubelet/nodeinfocache/... > new.txt
# (checkout old code)
go test -bench=. -benchmem -count=10 ./pkg/kubelet/lifecycle/... > old.txt
benchstat old.txt new.txt
```

### Files to Modify Summary

| File | Changes |
|------|---------|
| `pkg/kubelet/nodeinfocache/cache.go` | **New file** - Cache implementation |
| `pkg/kubelet/nodeinfocache/cache_test.go` | **New file** - Unit tests |
| `pkg/kubelet/nodeinfocache/benchmark_test.go` | **New file** - Benchmarks |
| `pkg/kubelet/kubelet.go` | Add field, initialize, wire lifecycle updates |
| `pkg/kubelet/kubelet_nodecache.go` | Update cache on node changes |
| `pkg/kubelet/lifecycle/predicate.go` | Add interface, accept cache, use Snapshot() |
| `pkg/kubelet/lifecycle/predicate_test.go` | Update tests for new parameter |

### Note on `pkg/kubelet/cache/`

The `pkg/kubelet/cache/` directory contains **interim code** copied from the scheduler cache. It can be:
- **Replaced**: Use the simpler implementation above
- **Discarded**: The scheduler cache is overengineered for kubelet's needs

The scheduler cache is designed for cluster-wide scheduling with optimistic pod binding. The kubelet only needs single-node, actual-state caching - much simpler requirements.

## Alternative Approach: Hash-Based Cache in predicateAdmitHandler

This approach keeps the cache self-contained within `predicateAdmitHandler`, using hash-based invalidation to detect when the cache is stale.

### Cache Location

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

### Cache Key/Invalidation Strategy

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

### Implementation

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

### Limitations

- **Cache miss on every pod set change**: Sequential pod admissions will miss cache since each changes the pod set
- **Primary benefit is resize retries**: Same pod set evaluated repeatedly while waiting for resources
- **No benefit for normal admission flow**: Each admission adds a pod, changing the hash

### When to Prefer This Approach

- If distributed cache update wiring is undesirable
- If code locality is prioritized over optimization scope
- If stale data risk (however small) is unacceptable

## Alternative Approach: Cache at Allocation Manager Level

Since `canAdmitPod` and `canResizePod` both need NodeInfo, consider caching at `pkg/kubelet/allocation/allocation_manager.go` instead. This would:
- Centralize cache management
- Benefit both admission and resize flows
- Align with where `allocatedPods` is computed

However, this still requires wiring updates from pod lifecycle events, similar to the recommended approach.

## Expected Performance

Based on PR #134462 benchmarks:
- Cache hit: **O(1)** lookup vs **O(n×m)** iteration
- ~34.5 microseconds per admission with cache hits on 100-pod nodes

With the incremental approach:
- **Every admission benefits** (not just resize retries)
- Pod add/remove: O(1) for add, O(n) for remove
- Snapshot for admission: O(n) deep copy (but no resource iteration)

## Compatibility with Existing PR #134462

The existing PR takes a hash-based approach but has issues:
1. Contains merge commits (needs rebase)
2. Missing extended resource support in hash (per reviewer)
3. May be over-engineering invalidation triggers

The incremental approach sidesteps these issues entirely by not using hash-based invalidation.

## Related Files

- `pkg/kubelet/lifecycle/predicate.go` - Main admission handler with NodeInfo construction
- `pkg/kubelet/allocation/allocation_manager.go` - Calls admission for pods and resizes
- `pkg/kubelet/kubelet_nodecache.go` - Existing node caching pattern to follow
- `pkg/scheduler/framework/types.go` - NodeInfo type with incremental update methods
- `pkg/scheduler/backend/cache/` - Scheduler cache pattern (more complex, for reference)

## References

- Issue: https://github.com/kubernetes/kubernetes/issues/132858
- Related PR: https://github.com/kubernetes/kubernetes/pull/134462
