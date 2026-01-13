# Package: config

## Purpose
Defines the internal configuration types for the PersistentVolume controller used by the kube-controller-manager.

## Key Types

- **PersistentVolumeBinderControllerConfiguration**: Main configuration struct containing:
  - `PVClaimBinderSyncPeriod`: Period for syncing PVs and PVCs.
  - `VolumeConfiguration`: Nested volume configuration.

- **VolumeConfiguration**: Configuration for volume-related features:
  - `EnableHostPathProvisioning`: Enable HostPath provisioning (testing only).
  - `EnableDynamicProvisioning`: Enable dynamic volume provisioning (default: true).
  - `PersistentVolumeRecyclerConfiguration`: Recycler pod configuration.
  - `FlexVolumePluginDir`: Directory for FlexVolume plugins.

- **PersistentVolumeRecyclerConfiguration**: Recycler settings:
  - `MaximumRetry`: Retry count for recycler operations.
  - `MinimumTimeoutNFS/IncrementTimeoutNFS`: Timeout settings for NFS recycler.
  - `PodTemplateFilePathNFS`: Custom NFS recycler pod template.
  - `MinimumTimeoutHostPath/IncrementTimeoutHostPath`: HostPath recycler timeouts.
  - `PodTemplateFilePathHostPath`: Custom HostPath recycler pod template.

## Design Notes

- HostPath provisioning is for testing only; it does not work in multi-node clusters.
- Recycler timeouts are calculated as: minimum + (incrementPerGi * volumeSizeGi).
- Part of the component config pattern used across Kubernetes controllers.
