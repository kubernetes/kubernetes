# Package: persistentvolume

## Purpose
Implements the PersistentVolume controller that manages the lifecycle of PersistentVolumes (PVs) and PersistentVolumeClaims (PVCs), including binding, provisioning, recycling, and deletion operations.

## Key Types

- **PersistentVolumeController**: Main controller managing PV/PVC binding and lifecycle.
- **ControllerParameters**: Input configuration for creating the controller.
- **persistentVolumeOrderedIndex**: Custom index for efficient PV lookups by access modes and storage class.

## Key Functions

### Controller Lifecycle
- **NewController(ctx, params)**: Creates a new PV controller instance.
- **Run(ctx)**: Starts the controller with volume and claim workers.
- **initializeCaches**: Populates internal caches from informers at startup.

### Volume Operations (pv_controller.go)
- **syncVolume(ctx, volume)**: Main state machine for volume lifecycle.
- **syncClaim(ctx, claim)**: Main state machine for claim lifecycle.
- **syncBoundClaim/syncUnboundClaim**: Handle claims in different phases.
- **bindVolumeToClaim**: Creates bidirectional binding between PV and PVC.
- **provisionClaim**: Triggers dynamic provisioning for a claim.
- **deleteVolumeOperation**: Handles volume deletion/recycling.
- **reclaimVolume**: Implements reclaim policy (Retain/Delete/Recycle).

### Indexing (index.go)
- **findBestMatchForClaim**: Finds smallest PV that satisfies claim requirements.
- **accessModesIndexFunc**: Indexes volumes by access modes for efficient matching.

## Design Notes

- Uses two separate work queues: one for volumes, one for claims.
- Maintains internal caches (volumes store, claims store) for version checking.
- Volume-claim binding uses annotations to track controller-managed bindings.
- Supports CSI migration with annotation updates for migrated volumes.
- Implements HonorPVReclaimPolicy feature for deletion protection finalizers.
- Periodic resync ensures eventual consistency.
