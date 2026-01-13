# Package: config

## Purpose
Defines the internal configuration types for the ValidatingAdmissionPolicyStatus controller used by the kube-controller-manager.

## Key Types

- **ValidatingAdmissionPolicyStatusControllerConfiguration**: Configuration struct containing:
  - `ConcurrentPolicySyncs`: Number of policy objects allowed to sync concurrently (default: 5). Higher values provide faster type checking but increase CPU and network load.

## Design Notes

- Uses `+k8s:deepcopy-gen=package` for automatic deep copy generation.
- This is the internal representation; the external v1alpha1 version is in the config/v1alpha1 subpackage.
- Part of the component config pattern used across Kubernetes controllers.
