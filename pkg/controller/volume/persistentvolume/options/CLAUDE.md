# Package: options

## Purpose
Defines CLI flag structures and defaults for the PersistentVolume controller configuration in kube-controller-manager.

## Key Types

- **VolumeConfigFlags**: Contains all CLI flags for configuring volume plugins:
  - `PersistentVolumeRecyclerMaximumRetry`: Max retry count for recycler (default: 3)
  - `PersistentVolumeRecyclerMinimumTimeoutNFS/HostPath`: Base timeout for recyclers
  - `PersistentVolumeRecyclerIncrementTimeoutNFS/HostPath`: Per-Gi timeout increment
  - `PersistentVolumeRecyclerPodTemplateFilePathNFS/HostPath`: Custom recycler pod templates
  - `EnableHostPathProvisioning`: Enable HostPath provisioner (testing only)
  - `EnableDynamicProvisioning`: Enable dynamic provisioning (default: true)

- **PersistentVolumeControllerOptions**: Main options struct containing:
  - `PVClaimBinderSyncPeriod`: Sync period for PVs and PVCs (default: 15s)
  - `VolumeConfigFlags`: Embedded volume configuration flags

## Key Functions

- **NewPersistentVolumeControllerOptions()**: Creates options with default values.
- **AddFlags(fs)**: Registers all CLI flags with the provided pflag.FlagSet.

## Design Notes

- HostPath provisioning is explicitly for development/testing only and won't work in multi-node clusters.
- The controller manager binary uses these flags to configure appropriate volume plugin instances.
- Recycler timeouts are calculated as: minimum + (incrementPerGi * volumeSizeGi).
