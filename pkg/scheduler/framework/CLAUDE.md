# Package: framework

## Purpose
Defines the core scheduling framework interfaces, types, and utilities. This package contains the extension points and data structures that plugins implement and the scheduler uses to make scheduling decisions.

## Key Types

### Framework (interface)
The main interface for the scheduling framework. Extends `fwk.Handle` with methods for:
- Running plugin extension points (PreFilter, Filter, PostFilter, Score, Reserve, Permit, PreBind, Bind, PostBind)
- Queue management (QueueSortFunc, PreEnqueuePlugins)
- Pod signing for result caching (SignPod, GetNodeHint, StoreScheduleResults)
- Plugin introspection (HasFilterPlugins, HasScorePlugins, ListPlugins)

### NodeToStatus
Maps node names to scheduling statuses. Tracks why pods can't be scheduled on specific nodes with support for "absent nodes" (default status for unlisted nodes).

### CycleState
Thread-safe key-value store for sharing data between plugins during a scheduling cycle. Uses sync.Map for concurrent access with support for:
- Read/Write/Delete operations
- Cloning for parallel operations
- Skip plugin tracking

### NodeInfo (types.go:165-208)

Aggregates node and pod information for scheduling decisions. Contains:
- `node *v1.Node` - The node object
- `Pods []fwk.PodInfo` - All pods on the node
- `Requested *Resource` - Total requested resources of all pods
- `Allocatable *Resource` - Node's allocatable resources
- `UsedPorts fwk.HostPortInfo` - Ports allocated on the node
- `Generation int64` - Bumped on every change for cache invalidation

**Thread Safety**: NodeInfo is **NOT thread-safe**. Requires external synchronization (typically via `sync.RWMutex` in a cache wrapper).

#### Incremental Update Methods

| Method | Line | Complexity | Description |
|--------|------|------------|-------------|
| `AddPod(pod)` | 359 | O(1) | Adds pod, updates resource totals |
| `AddPodInfo(podInfo)` | 347 | O(1) | Adds pre-computed PodInfo |
| `RemovePod(logger, pod)` | 403 | O(n) | Finds and removes pod by UID |
| `SetNode(node)` | 477 | O(1) | Updates node metadata and allocatable |
| `RemoveNode()` | 487 | O(1) | Clears node reference, keeps pods |
| `Snapshot()` | - | O(n) | Returns deep copy for concurrent use |

#### Internal update() Method (line 425)

Called by `AddPodInfo` (sign=+1) and `RemovePod` (sign=-1):
- Updates `Requested` resources (CPU, memory, ephemeral storage, scalars)
- Updates `NonZeroRequested` (minimum values per container)
- Updates `UsedPorts` via `updateUsedPorts()`
- Updates `PVCRefCounts` via `updatePVCRefCounts()`
- Bumps `Generation` counter

#### Usage Pattern

The scheduler cache wraps NodeInfo with mutex protection:
```go
// In scheduler cache
cache.mu.Lock()
nodeInfo.AddPod(pod)  // Safe under lock
cache.mu.Unlock()

// For reads, take snapshot
snapshot := nodeInfo.Snapshot()  // Deep copy for concurrent use
```

Both scheduler (`pkg/scheduler/backend/cache/`) and kubelet use NodeInfo for resource tracking. The kubelet currently rebuilds NodeInfo on every admission (issue #132858).

### PodsToActivate
State data for tracking pods that should be moved to activeQ. Used by plugins to trigger requeueing of related pods.

## Key Functions

- **NewCycleState()**: Creates a new cycle state
- **NewDefaultNodeToStatus()**: Creates NodeToStatus with default "UnschedulableAndUnresolvable" for absent nodes
- **PodSchedulingPropertiesChange(newPod, oldPod)**: Interprets pod updates and returns relevant cluster events
- **NodeSchedulingPropertiesChange(newNode, oldNode)**: Interprets node updates and returns relevant cluster events

## Event Constants
- **ScheduleAttemptFailure**: When scheduling fails
- **BackoffComplete**: When a pod finishes backoff
- **ForceActivate**: When a pod is forcibly moved to activeQ
- **UnschedulableTimeout**: When a pod times out in unschedulable queue

## Design Pattern
- Plugin-based architecture with well-defined extension points
- Event-driven requeueing based on cluster changes
- State sharing via CycleState with cloneable data
