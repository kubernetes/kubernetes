# Package cache

Package cache provides the scheduler's internal cache for tracking pod and node state, enabling efficient scheduling decisions with optimistic pod assumptions.

## Key Types

- `Cache`: Interface defining the scheduler cache operations for pods and nodes
- `cacheImpl`: Concrete implementation with doubly-linked list for node ordering by update time
- `podState`: Tracks pod with expiration deadline and binding status
- `nodeInfoListItem`: Doubly-linked list node for NodeInfo ordering
- `Dump`: Snapshot of cache state for debugging
- `Snapshot`: Point-in-time view of cache for scheduling cycles

## Key Functions

- `New(ctx, ttl)`: Creates a new cache with automatic assumed pod expiration
- `AssumePod`: Optimistically adds a pod before binding confirmation
- `FinishBinding`: Marks assumed pod as ready for expiration
- `ForgetPod`: Removes an assumed pod (e.g., on binding failure)
- `AddPod/UpdatePod/RemovePod`: Standard pod lifecycle operations
- `AddNode/UpdateNode/RemoveNode`: Node lifecycle operations
- `UpdateSnapshot`: Efficiently updates a snapshot for scheduling cycles
- `BindPod`: Handles async pod binding via API dispatcher

## State Machine

Pods flow through states: Initial -> Assumed -> Added -> Deleted/Expired
- Assumed pods can expire if binding confirmation is not received
- "Forget" transitions assumed pods back to initial state

## Design Notes

- Uses generation numbers for efficient incremental snapshot updates
- Maintains node image states for image locality scoring
- Ghost nodes kept when pods remain after node deletion
- Thread-safe with RWMutex protection
