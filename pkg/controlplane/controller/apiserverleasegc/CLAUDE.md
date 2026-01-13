# Package: apiserverleasegc

## Purpose
This package implements a controller that garbage collects expired API server identity leases. It cleans up leases from API servers that have stopped running or crashed without proper cleanup.

## Key Types

- **Controller**: The GC controller that watches and deletes expired leases. Contains a filtered lease informer scoped to a specific namespace and label selector

## Key Functions

- **NewAPIServerLeaseGC()**: Creates a new GC controller with a filtered informer for a specific namespace and label selector
- **Run() / RunWithContext()**: Starts the controller - runs the informer and periodically invokes garbage collection
- **gc()**: Lists all leases, checks if expired (both from cache and API server), and deletes expired ones
- **isLeaseExpired()**: Checks if a lease has expired based on RenewTime + LeaseDurationSeconds

## Design Notes

- Uses a filtered informer that only watches leases in a specific namespace with a specific label selector
- Double-checks lease expiration against the API server before deletion to avoid race conditions in HA clusters
- Handles the case where another GC controller in the HA cluster has already deleted the lease
- Leases are considered expired if RenewTime is nil, LeaseDurationSeconds is nil, or RenewTime + duration is in the past
