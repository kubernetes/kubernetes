# Package: ephemeral

## Purpose
Implements the Ephemeral Volume controller that automatically creates PersistentVolumeClaims for pods using ephemeral inline volumes (volumes with `ephemeral` volume source).

## Key Types

- **Controller**: Interface with Run(ctx, workers) method.
- **ephemeralController**: Implementation that watches pods and creates PVCs.

## Key Functions

- **NewController(ctx, kubeClient, podInformer, pvcInformer)**: Creates a new ephemeral controller.
- **Run(ctx, workers)**: Starts the controller workers.
- **enqueuePod(obj)**: Enqueues pods that have ephemeral volumes (skips pods being deleted).
- **onPVCDelete(obj)**: Re-enqueues pods when their ephemeral PVCs are deleted (to recreate them).
- **syncHandler(ctx, key)**: Processes a pod, creating PVCs for each ephemeral volume.
- **handleVolume(ctx, pod, vol)**: Creates a PVC for a single ephemeral volume if it does not exist.

## PVC Creation Details

- PVC name is derived from pod name + volume name via `ephemeral.VolumeClaimName()`.
- PVC has owner reference pointing to the pod (enables garbage collection).
- PVC spec comes from the volume's `VolumeClaimTemplate`.
- Labels and annotations are copied from the template.

## Design Notes

- Pod spec is immutable, so pod updates are ignored.
- PVC deletion triggers pod re-processing to recreate the PVC.
- Uses `ephemeral.VolumeIsForPod()` to verify PVC ownership before skipping creation.
- Records events on binding failures.
- Tracks create attempts and failures via metrics.
