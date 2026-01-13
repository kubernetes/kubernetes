# Package: sync

Synchronizes node IP address management (IPAM) state between Kubernetes nodes and cloud provider aliases.

## Key Types

- **NodeSync**: Manages CIDR synchronization for a single node, coordinating between cloud aliases and Kubernetes node specs.
- **NodeSyncMode**: Defines sync direction - `SyncFromCloud` (cloud -> node) or `SyncFromCluster` (node -> cloud).
- **cloudAlias**: Interface for cloud platform IP alias operations.
- **kubeAPI**: Interface for Kubernetes API operations on nodes.

## Key Functions

- **New**: Creates a new NodeSync instance for a given node.
- **Loop**: Runs the sync loop, processing update/delete operations and periodic resyncs.
- **Update**: Triggers an update operation to sync CIDR allocation.
- **Delete**: Handles node deletion and releases the CIDR range.

## Design Patterns

- Uses channel-based operation dispatch (`opChan`) for thread-safe sync operations.
- Implements updateOp and deleteOp as separate operation types following the command pattern.
- Handles multiple sync scenarios: allocate new range, sync from cloud, sync to cloud, validate existing.
- Emits Kubernetes events for error conditions (InvalidPodCIDR, InvalidModeEvent, MismatchEvent).
