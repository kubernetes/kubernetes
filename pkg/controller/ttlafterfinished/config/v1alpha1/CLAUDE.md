# Package: v1alpha1

## Purpose
Provides versioned (v1alpha1) configuration types, defaults, and conversion functions for the TTL After Finished controller configuration.

## Key Functions

- **RecommendedDefaultTTLAfterFinishedControllerConfiguration(obj)**: Sets recommended defaults:
  - `ConcurrentTTLSyncs`: defaults to 5 if not set or <= 0
- **Convert_v1alpha1_TTLAfterFinishedControllerConfiguration_To_config_TTLAfterFinishedControllerConfiguration**: Converts from v1alpha1 to internal config type.
- **Convert_config_TTLAfterFinishedControllerConfiguration_To_v1alpha1_TTLAfterFinishedControllerConfiguration**: Converts from internal config to v1alpha1.

## Key Variables

- **SchemeBuilder**: Runtime scheme builder for registering types.
- **AddToScheme**: Function to register this API group/version to a scheme.

## Design Notes

- Uses code generation tags for deep copy and conversion generation.
- External types are defined in `k8s.io/kube-controller-manager/config/v1alpha1`.
- Defaults are intentionally not registered in the scheme to allow consumers to opt-out.
