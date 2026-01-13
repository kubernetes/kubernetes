# Package: reconciler

## Purpose
Implements the reconciliation loop that compares the desired state of world with the actual state of world and triggers attach/detach operations to bring them into alignment.

## Key Types

- **Reconciler**: Interface with Run(ctx) method.
- **reconciler**: Implementation containing DSW, ASW, operation executor, and configuration.

## Key Functions

- **NewReconciler(...)**: Creates a new reconciler with configurable timing and behavior.
- **Run(ctx)**: Starts the reconciliation loop.
- **reconcile(ctx)**: Main reconciliation logic - processes detaches then attaches.
- **attachDesiredVolumes**: Triggers attach operations for volumes in DSW not in ASW.
- **syncStates**: Verifies attached volumes are still attached via storage provider.
- **reportMultiAttachError**: Reports events when multi-attach is attempted on non-multi-attach volumes.

## Key Behaviors

### Detach Logic
- Processes detaches before attaches (for pod rescheduling).
- Checks if volume is mounted before detaching.
- Force detaches from unhealthy nodes after maxWaitForUnmountDuration.
- Respects `node.kubernetes.io/out-of-service` taint for immediate force detach.
- Updates node status before triggering detach.

### Attach Logic
- Skips if operation already pending.
- Resets detach request time for already-attached volumes.
- Reports multi-attach errors for non-multi-attach volumes.

## Design Notes

- Distinct from kubelet's volume manager reconciler (different scope).
- Configurable: loopPeriod, maxWaitForUnmountDuration, syncDuration.
- Optional features: disableReconciliationSync, disableForceDetachOnTimeout.
- Records metrics for force detach operations.
