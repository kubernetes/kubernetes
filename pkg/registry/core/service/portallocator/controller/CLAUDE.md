# Package: controller

## Purpose
Provides the NodePort repair controller that periodically verifies and repairs Service port allocations across the cluster.

## Key Types

- **Repair**: Controller that runs periodic repair loops for NodePort allocations.

## Key Functions

- **NewRepair(interval, serviceClient, eventClient, portRange, alloc)**: Creates repair controller.
- **RunUntil(onFirstSuccess, stopCh)**: Starts the controller loop.
- **runOnce()**: Verifies allocation state with retry on conflict.
- **doRunOnce()**: Main repair logic:
  - Gets current allocation snapshot from etcd
  - Lists all Services and rebuilds expected allocation state
  - Detects and reports: duplicates, out-of-range ports, unallocated ports
  - Handles leaked ports (waits numRepairsBeforeLeakCleanup before cleanup)
  - Persists corrected allocation state
- **collectServiceNodePorts(service)**: Extracts all NodePorts from Service including HealthCheckNodePort.

## Design Notes

- Mirror of ipallocator/controller/repair.go for ports.
- Emits events for allocation errors (PortNotAllocated, PortAlreadyAllocated, PortOutOfRange).
- Leak detection uses counter to avoid races between allocation and use.
- Handles same NodePort with different protocols (deduplicates).
- Level-driven and idempotent - rebuilds complete state each run.
