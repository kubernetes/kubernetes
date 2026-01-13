# Package: config

## Purpose
Defines the internal configuration types for the TTL After Finished controller used by the kube-controller-manager.

## Key Types

- **TTLAfterFinishedControllerConfiguration**: Configuration struct containing:
  - `ConcurrentTTLSyncs`: Number of concurrent TTL-after-finished collector workers allowed to sync.

## Design Notes

- Uses `+k8s:deepcopy-gen=package` for automatic deep copy generation.
- This is the internal representation; the external v1alpha1 version is in the config/v1alpha1 subpackage.
- Part of the component config pattern used across Kubernetes controllers.
