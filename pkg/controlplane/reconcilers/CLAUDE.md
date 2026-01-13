# Package: reconcilers

## Purpose
This package provides endpoint reconciler implementations for managing the "kubernetes" service endpoints in the default namespace. It ensures that the service endpoints accurately reflect the active API server instances in the cluster.

## Key Types

- **EndpointReconciler**: Interface for reconciling API server service endpoints
- **Leases**: Interface for managing master IP leases in etcd storage
- **leaseEndpointReconciler**: Lease-based reconciler using etcd TTL keys
- **masterCountEndpointReconciler**: Static master count-based reconciler
- **noneEndpointReconciler**: No-op reconciler that disables endpoint management
- **EndpointsAdapter**: Adapter for managing Endpoints and EndpointSlice resources

## Key Functions

- **NewLeaseEndpointReconciler()**: Creates a lease-based reconciler using etcd storage leases
- **NewMasterCountEndpointReconciler()**: Creates a reconciler based on expected master count
- **NewNoneEndpointReconciler()**: Creates a no-op reconciler
- **NewLeases()**: Creates an etcd-based lease manager with TTL support
- **ReconcileEndpoints()**: Updates endpoints to match active API servers
- **RemoveEndpoints()**: Removes this server's endpoint on shutdown

## Reconciler Types

- **lease**: Uses etcd TTL keys to track active masters (recommended for HA)
- **master-count**: Uses a static count of expected masters
- **none**: Disables endpoint reconciliation entirely

## Design Notes

- Lease reconciler stores IP leases with TTL in etcd under a base key
- Master count reconciler maintains exactly N endpoints, removing extras lexicographically
- Sets skip-mirror annotation to prevent EndpointSliceMirroring controller conflicts
- Thread-safe with mutex protection during reconciliation
- Supports graceful shutdown by removing endpoints before stopping
