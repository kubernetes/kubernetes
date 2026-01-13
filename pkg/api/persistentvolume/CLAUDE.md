# Package: persistentvolume

## Purpose
Provides utilities for PersistentVolume API objects including feature-gated field handling and deprecation warnings.

## Key Functions
- `DropDisabledSpecFields(pvSpec *api.PersistentVolumeSpec, oldPVSpec *api.PersistentVolumeSpec)` - Removes feature-gated fields (VolumeAttributesClassName) when the VolumeAttributesClass feature is disabled
- `GetWarningsForPersistentVolume(pv *api.PersistentVolume) []string` - Generates all warnings for a PV

## Warnings Generated
- Deprecated annotations: `volume.beta.kubernetes.io/storage-class` (since v1.8), `volume.beta.kubernetes.io/mount-options` (since v1.31)
- Deprecated reclaim policy: `Recycle` - recommends dynamic provisioning instead
- Deprecated node labels in nodeAffinity selectors
- Deprecated/non-functional volume plugins:
  - CephFS (deprecated v1.28, non-functional v1.31+)
  - PhotonPersistentDisk (deprecated v1.11, non-functional v1.16+)
  - ScaleIO (deprecated v1.16, non-functional v1.22+)
  - StorageOS (deprecated v1.22, non-functional v1.25+)
  - Glusterfs (deprecated v1.25, non-functional v1.26+)
  - RBD (deprecated v1.28, non-functional v1.31+)
