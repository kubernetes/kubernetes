# Package: selinuxwarning

## Purpose
Implements an optional controller that emits warning events and metrics when two pods might use the same volume with different SELinux labels, which could cause access issues on SELinux-enabled clusters.

## Key Types

- **Controller**: Main controller watching Pods, PVCs, PVs, and CSIDrivers to detect SELinux label conflicts.

## Key Functions

### Controller Lifecycle
- **NewController(ctx, kubeClient, informers...)**: Creates the controller with required informers and volume plugins.
- **Run(ctx, workers)**: Starts the controller workers.

### Event Handlers
- **enqueuePod/updatePod**: Queue pods for processing; only reacts to terminal phase changes.
- **addPVC/updatePVC**: Queue pods when their PVCs become bound.
- **addPV/updatePV**: Queue pods when PVs are bound to their PVCs.
- **addCSIDriver/updateCSIDriver/deleteCSIDriver**: Queue pods when CSIDriver SELinuxMount capability changes.

### Sync Logic
- **sync(ctx, podRef)**: Main sync handler for a pod.
- **syncPod(ctx, pod)**: Processes all volumes in a pod, checking for SELinux conflicts.
- **syncVolume(logger, pod, spec, seLinuxLabel, ...)**: Adds volume to cache and reports any conflicts.
- **syncPodDelete(ctx, podRef)**: Removes pod's volumes from cache.

## Key Subpackages

- **cache**: Implements VolumeCache for tracking volume-to-SELinux-label mappings.
- **translator**: SELinux label translation utilities.

## Design Notes

- Does nothing on clusters without SELinux.
- Does not modify API objects except for warning events.
- Reports conflicts even for pods that may never run on the same node.
- Respects SELinuxChangePolicy: Recursive policy pods don't conflict.
- Uses PodPVCIndex for efficient PVC-to-pod lookups.
- Tracks metrics for SELinux label conflicts.
