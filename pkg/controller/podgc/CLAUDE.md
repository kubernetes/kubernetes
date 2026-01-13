# Package: podgc

Implements the Pod Garbage Collection controller that cleans up terminated, orphaned, and terminating pods.

## Key Types

- **PodGCController**: Controller that periodically garbage collects pods based on various conditions.

## Key Functions

- **NewPodGC**: Creates a new controller with the specified terminated pod threshold.
- **Run**: Starts the controller, running gc periodically (every 20 seconds by default).
- **gc**: Main GC logic that runs all collection strategies.
- **gcTerminated**: Deletes terminated pods exceeding the threshold, oldest first.
- **gcTerminating**: Deletes terminating pods on nodes with out-of-service taint.
- **gcOrphaned**: Deletes pods assigned to non-existent nodes (after quarantine period).
- **gcUnscheduledTerminating**: Deletes terminating pods that were never scheduled.

## Key Constants

- **gcCheckPeriod**: 20 seconds between GC runs.
- **quarantineTime**: 40 seconds before confirming node deletion for orphan detection.

## Design Patterns

- Uses quarantine period to avoid race conditions with node informer updates.
- Sorts terminated pods by eviction status and creation time (evicted first, then oldest).
- Marks pods as Failed phase before deletion for proper status reporting.
- Adds DisruptionTarget condition before deleting orphaned pods.
- Exposes metrics for pod deletions and errors by reason.
