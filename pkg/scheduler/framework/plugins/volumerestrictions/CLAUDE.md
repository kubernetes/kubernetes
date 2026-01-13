# Package: volumerestrictions

## Purpose
Implements volume restriction checks for scheduling. Prevents scheduling pods that would violate volume access mode constraints, particularly ReadWriteOncePod.

## Key Types

### VolumeRestrictions
The plugin struct implementing:
- PreFilterPlugin, FilterPlugin, EnqueueExtensions, SignPlugin
- **pvcLister**: For PVC access mode lookup
- **sharedLister**: For checking existing pod volumes

### preFilterState
Per-pod state tracking:
- **readWriteOncePodPVCs**: Names of RWO-Pod PVCs used by the pod
- **conflictingPVCRefCount**: Count of conflicts with scheduled pods

## Extension Points

### PreFilter
- Identifies ReadWriteOncePod PVCs used by the pod
- Counts existing references to these PVCs across the cluster
- Skips if no RWO-Pod volumes (returns Skip status)

### PreFilterExtensions
- AddPod/RemovePod: Updates conflict count when pods are assumed/removed
- Enables incremental updates during scheduling

### Filter
- Checks for disk conflicts (legacy NoDiskConflict)
- Verifies ReadWriteOncePod volumes aren't already in use on the node
- Returns Unschedulable if conflicts detected

## ReadWriteOncePod Access Mode
- PVC can only be used by a single pod at a time
- More restrictive than ReadWriteOnce (single node)
- Used for workloads requiring exclusive volume access

## Key Functions

- **New(ctx, obj, handle, features)**: Creates the plugin
- **SignPod(ctx, pod)**: Returns volumes for pod signing
- **EventsToRegister()**: Returns PVC/Add, PVC/Update, Pod/Delete events
- **haveOverlap(a, b []string)**: Checks for volume name conflicts

## Design Pattern
- Uses incremental state updates for efficiency
- Supports both legacy disk conflict and new RWO-Pod checks
- Queueing hints respond to PVC and pod changes
