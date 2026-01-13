# Package: util

## Purpose
Provides utility functions for volume spec creation and processing in the attach/detach controller, including PVC/PV dereferencing and CSI migration translation.

## Key Functions

### Volume Spec Creation
- **CreateVolumeSpec(logger, podVolume, pod, ...)**: Creates a volume.Spec from a pod volume, dereferencing PVC/PV and translating to CSI if needed.
- **CreateVolumeSpecWithNodeMigration(logger, podVolume, pod, nodeName, ...)**: Same as above but checks node-specific CSI migration support.
- **createInTreeVolumeSpec**: Internal function that handles PVC/ephemeral volume dereferencing.

### Pod Processing
- **ProcessPodVolumes(logger, pod, addVolumes, dsw, ...)**: Processes all volumes in a pod and adds/removes them from the desired state of world.
- **DetermineVolumeAction(pod, dsw, defaultAction)**: Returns whether to add (true) or remove (false) volumes based on pod state.

### Cache Helpers
- **getPVCFromCache(namespace, name, pvcLister)**: Fetches bound PVC from cache.
- **getPVSpecFromCache(name, pvcReadOnly, expectedClaimUID, pvLister)**: Fetches PV and creates volume.Spec.

### CSI Migration
- **translateInTreeSpecToCSIIfNeeded**: Translates in-tree spec to CSI unconditionally if migratable.
- **translateInTreeSpecToCSIOnNodeIfNeeded**: Translates only if migration is supported on the specific node.
- **isCSIMigrationSupportedOnNode**: Checks CSINode annotations for migration support.

## Design Notes

- Handles both inline volumes and PVC-backed volumes.
- Supports ephemeral volumes (verifies PVC ownership).
- Deep copies PV objects to prevent cache mutation.
- Skips non-attachable volumes and terminated pods.
- CSI migration check uses CSINode's MigratedPluginsAnnotationKey.
