# Package: v1alpha1

## Purpose
Provides versioned (v1alpha1) configuration types, defaults, and conversion functions for the Attach/Detach controller configuration.

## Key Functions

- **RecommendedDefaultAttachDetachControllerConfiguration(obj)**: Sets recommended defaults:
  - `ReconcilerSyncLoopPeriod`: defaults to 60 seconds if not set.

## Design Notes

- Uses code generation tags for deep copy and conversion generation.
- External types are defined in `k8s.io/kube-controller-manager/config/v1alpha1`.
- Defaults are intentionally not registered in the scheme to allow consumers to opt-out.
- DisableAttachDetachReconcilerSync and DisableForceDetachOnTimeout default to false (features enabled).
