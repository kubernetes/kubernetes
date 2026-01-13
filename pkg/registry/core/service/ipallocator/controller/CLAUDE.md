# Package: controller

## Purpose
Provides the ClusterIP repair controller that periodically verifies and repairs Service IP allocations across the cluster.

## Key Types

- **Repair**: Controller that runs periodic repair loops for ClusterIP allocations.

## Key Functions

- **NewRepair(interval, serviceClient, eventClient, network, alloc, secondaryNetwork, secondaryAlloc)**: Creates repair controller for single or dual-stack clusters.
- **RunUntil(onFirstSuccess, stopCh)**: Starts the controller loop.
- **runOnce()**: Verifies allocation state with retry on conflict.
- **doRunOnce()**: Main repair logic:
  - Gets current allocation snapshots from etcd
  - Lists all Services and rebuilds expected allocation state
  - Detects and reports: duplicates, out-of-range IPs, unallocated IPs
  - Handles leaked IPs (waits numRepairsBeforeLeakCleanup before cleanup)
  - Persists corrected allocation state

## Design Notes

- Handles dual-stack clusters with separate allocators per IP family.
- Emits events for allocation errors (ClusterIPNotValid, ClusterIPAlreadyAllocated, ClusterIPOutOfRange).
- Leak detection uses counter to avoid races between allocation and use.
- Level-driven and idempotent - rebuilds complete state each run.
- Metrics for repair errors by type.
