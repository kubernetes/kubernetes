# Package: cache

## Purpose
Implements the scheduler's internal cache for maintaining cluster state. The cache is the heart of Kubernetes scheduling decisions, providing an efficient, in-memory representation of cluster state with optimistic pod scheduling support.

## Core Data Structures

### cacheImpl (struct)
```go
type cacheImpl struct {
    mu              sync.RWMutex
    assumedPods     sets.Set[string]           // Pods assumed bound but not confirmed
    podStates       map[string]*podState       // Pod key -> state (deadline, bindingFinished)
    nodes           map[string]*nodeInfoListItem  // Node name -> linked list item
    headNode        *nodeInfoListItem          // Most recently updated node (list head)
    nodeTree        *nodeTree                  // Zone-aware node organization
    imageStates     map[string]*ImageStateSummary
    apiDispatcher   fwk.APIDispatcher          // Async API call handler
}
```

### podState (struct)
- `pod`: The actual pod object
- `deadline`: Expiration time for assumed pods (nil = never expires)
- `bindingFinished`: Flag to prevent expiration while binding is in progress

### nodeInfoListItem
Forms a doubly-linked list for tracking recently-updated nodes:
- Maintains pointer to `framework.NodeInfo` with aggregated node state
- Enables O(1) head updates and efficient snapshot generation

### Snapshot
Point-in-time view of cluster state for scheduling cycles:
```go
type Snapshot struct {
    nodeInfoMap                                  map[string]*framework.NodeInfo
    nodeInfoList                                 []fwk.NodeInfo  // All nodes, tree-ordered
    havePodsWithAffinityNodeInfoList             []fwk.NodeInfo  // Subset with affinity pods
    havePodsWithRequiredAntiAffinityNodeInfoList []fwk.NodeInfo
    usedPVCSet                                   sets.Set[string]
    generation                                   int64  // Tracks changes
}
```

## Pod State Machine

```
                      Add
Initial -> Assumed ──────────-> Added -> (Updated) -> Deleted
             |     ^               ^
             |     |               |
          Expire  Forget        Remove
             |     |               |
             v     |               |
           Expired ────────────────┘
```

**State Transitions:**
- **Assume**: Scheduler assumes pod is bound before API confirmation (optimistic scheduling)
- **Add**: Pod confirmed by API watcher; if already assumed, updates it; if expired, re-adds it
- **Update**: Only valid for non-assumed pods; removes old pod data and adds new data
- **Remove/Forget**: Deletes pod from cache
- **Expire**: Assumed pods that don't receive Add event within TTL are automatically expired

## Key Functions

### Cache Creation
- **New(ctx, ttl, apiDispatcher)**: Creates cache with assumed pod TTL and starts cleanup goroutine

### Pod Operations
- **AssumePod**: Temporarily adds pod before binding confirmed (used at scheduling decision point)
- **FinishBinding**: Sets deadline for assumed pod expiration, sets bindingFinished flag
- **ForgetPod**: Removes assumed pod (e.g., on scheduling failure)
- **AddPod**: Handles confirmed pod from API (updates assumed, re-adds expired, or adds new)
- **UpdatePod**: Updates non-assumed pod (removes old, adds new)
- **RemovePod**: Removes pod from node's pod list

### Node Operations
- **AddNode**: Creates new node entry, updates imageStates, moves to head
- **UpdateNode**: Handles zone changes via nodeTree, updates image states
- **RemoveNode**: Marks node as deleted but keeps it if pods remain (ghost nodes)

### Snapshot
- **UpdateSnapshot**: Creates consistent view for scheduling cycle using generation-based incremental updates

## Optimistic Scheduling

The cache supports "assume" semantics for reduced scheduling latency:
1. When scheduler decides to place a pod, it immediately updates the cache *before* the API call
2. Subsequent scheduling decisions see the assumed pod
3. TTL-based expiration cleans up if binding never completes
4. `FinishBinding` is called after API call to start the expiration timer

## Doubly-Linked List for Efficiency

Nodes are organized in a doubly-linked list ordered by update time:
- **Head**: Most recently updated nodes
- **Tail**: Least recently updated nodes
- Enables O(changed nodes) snapshot updates instead of O(all nodes)

## Snapshot Mechanism

The `UpdateSnapshot` algorithm:
1. **Generation-based optimization**: Compares snapshot generation with node generations
2. **Incremental updates**: Walks from headNode backward, stopping when generation <= snapshot generation
3. **Lazy full rebuilds**: Recalculates affinity lists only when nodes change
4. **Consistency validation**: Verifies snapshot node count matches nodeTree

## Zone-Aware Node Tree

The `nodeTree` organizes nodes by zone for locality-aware scheduling:
- Maps zones (region-zone pairs) to node arrays
- Supports round-robin listing across zones
- Thread-safe only with external synchronization (cache lock)

## Background Cleanup

A goroutine runs every 1 second (`cleanAssumedPeriod`) to:
- Remove expired assumed pods (deadline passed + bindingFinished = true)
- Clean up ghost nodes (deleted nodes with no remaining pods)
- Update cache size metrics

## Design Patterns

| Pattern | Purpose |
|---------|---------|
| Optimistic scheduling | Reduce latency by assuming before API confirms |
| Ghost nodes | Retain deleted nodes until all pods cleaned up |
| Generation tracking | Enable efficient incremental snapshot updates |
| Immutable snapshots | Safe concurrent use during scheduling cycle |
| TTL expiration | Prevent memory leaks from orphaned assumed pods |

## Thread Safety

- **RWMutex**: Guards all cache operations
- **Read locks**: For GetPod, IsAssumedPod, NodeCount, PodCount
- **Write locks**: For all modifications (Assume, Add, Update, Remove, node operations)
- **NodeTree**: NOT thread-safe itself; protected by cache RWMutex

## Consistency Checks

- **Node name validation**: Detects when pods move between nodes
- **Assumed state validation**: Prevents duplicate assumes, invalid transitions
- **Snapshot consistency**: Verifies snapshot state matches reality
- **Ghost node cleanup**: Removes deleted nodes when last pod leaves
