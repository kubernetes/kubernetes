# Package: config

## Purpose
Defines the internal configuration types for the Ephemeral Volume controller used by the kube-controller-manager.

## Key Types

- **EphemeralVolumeControllerConfiguration**: Configuration struct containing:
  - `ConcurrentEphemeralVolumeSyncs`: Number of ephemeral volume syncing operations done concurrently. Higher values = faster updates but more CPU/network load.

## Design Notes

- Uses `+k8s:deepcopy-gen=package` for automatic deep copy generation.
- This is the internal representation; the external v1alpha1 version is in the config/v1alpha1 subpackage.
- Part of the component config pattern used across Kubernetes controllers.
