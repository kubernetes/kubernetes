# Package: v1alpha1

## Purpose
Provides versioned (v1alpha1) configuration types, defaults, and conversion functions for the PersistentVolume controller configuration.

## Key Functions

- **RecommendedDefaultPersistentVolumeBinderControllerConfiguration(obj)**: Sets recommended defaults:
  - `PVClaimBinderSyncPeriod`: 15 seconds
  - Calls RecommendedDefaultVolumeConfiguration for nested config.

- **RecommendedDefaultVolumeConfiguration(obj)**: Sets volume defaults:
  - `EnableHostPathProvisioning`: false
  - `EnableDynamicProvisioning`: true
  - `FlexVolumePluginDir`: "/usr/libexec/kubernetes/kubelet-plugins/volume/exec/"

- **RecommendedDefaultPersistentVolumeRecyclerConfiguration(obj)**: Sets recycler defaults:
  - `MaximumRetry`: 3
  - `MinimumTimeoutNFS`: 300 seconds
  - `IncrementTimeoutNFS`: 30 seconds per Gi
  - `MinimumTimeoutHostPath`: 60 seconds
  - `IncrementTimeoutHostPath`: 30 seconds per Gi

## Design Notes

- External types are defined in `k8s.io/kube-controller-manager/config/v1alpha1`.
- Defaults are intentionally not registered in the scheme to allow consumers to opt-out.
- Uses `ptr.To()` for pointer field defaults.
