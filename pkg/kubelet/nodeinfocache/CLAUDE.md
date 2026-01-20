# Package: nodeinfocache

Thread-safe cache for `framework.NodeInfo` with incremental updates. Created for issue #132858 to avoid rebuilding NodeInfo on every admission.

## Key Types

- **Cache**: Thread-safe wrapper around `framework.NodeInfo`

## Key Functions

- `New()`: Creates empty cache
- `SetNode(node)`: Updates node metadata (O(1))
- `AddPod(pod)`: Adds pod incrementally (O(1))
- `RemovePod(logger, pod)`: Removes pod (O(n) - finds by UID)
- `UpdatePod(logger, old, new)`: Updates pod (remove + add)
- `Snapshot()`: Returns `*framework.NodeInfo` deep copy for concurrent use (O(n))
- `PodCount()`: Returns number of cached pods

**Important**: The cache's `Snapshot()` must use `nodeInfo.SnapshotConcrete()` internally (not `nodeInfo.Snapshot()`) to return the concrete `*framework.NodeInfo` type. This allows callers to call `SetNode()` on the returned snapshot. See `pkg/scheduler/framework/CLAUDE.md` for interface vs concrete type details.

## Thread Safety

- Uses `sync.RWMutex` for concurrent access
- `Snapshot()` returns a deep copy safe for concurrent use
- All mutating operations acquire write lock
- Read operations acquire read lock

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|------------|-------|
| AddPod | O(m) | m = containers in pod |
| RemovePod | O(n×m) | n = pods, m = containers |
| SetNode | O(1) | Just pointer assignment |
| Snapshot | O(n×m) | Deep copy of all state |

## Usage Pattern

```go
cache := nodeinfocache.New()

// Initialize with node
cache.SetNode(node)

// Add existing pods
for _, pod := range activePods {
    cache.AddPod(pod)
}

// Get snapshot for admission (returns *framework.NodeInfo)
nodeInfo := cache.Snapshot()
nodeInfo.SetNode(freshNode) // Safe - it's a deep copy, SetNode() available on concrete type
```

## Integration Points

- **Kubelet struct**: Holds the cache instance
- **predicateAdmitHandler**: Calls `Snapshot()` instead of `NewNodeInfo(pods...)`
- **HandlePodAdditions/Updates/Removes**: Call cache update methods
- **kubelet_nodecache.go**: Calls `SetNode()` when cached node updates

## Design Notes

- NodeInfo itself is NOT thread-safe, hence the wrapper
- Snapshot deep copy allows callers to modify without affecting cache
- RemovePod is O(n) due to UID-based lookup in slice
- Cache should be initialized with existing pods at kubelet startup
