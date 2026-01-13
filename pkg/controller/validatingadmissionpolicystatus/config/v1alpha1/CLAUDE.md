# Package: v1alpha1

## Purpose
Provides versioned (v1alpha1) configuration types, defaults, and conversion functions for the ValidatingAdmissionPolicyStatus controller configuration.

## Key Functions

- **RecommendedDefaultValidatingAdmissionPolicyStatusControllerConfiguration(obj)**: Sets recommended defaults:
  - `ConcurrentPolicySyncs`: defaults to 5 if set to 0
- **Convert_v1alpha1_ValidatingAdmissionPolicyStatusControllerConfiguration_To_config_ValidatingAdmissionPolicyStatusControllerConfiguration**: Converts from v1alpha1 to internal config type.
- **Convert_config_ValidatingAdmissionPolicyStatusControllerConfiguration_To_v1alpha1_ValidatingAdmissionPolicyStatusControllerConfiguration**: Converts from internal config to v1alpha1.

## Key Variables

- **SchemeBuilder**: Runtime scheme builder for registering types.
- **AddToScheme**: Function to register this API group/version to a scheme.

## Design Notes

- Uses code generation tags for deep copy and conversion generation.
- External types are defined in `k8s.io/kube-controller-manager/config/v1alpha1`.
- Defaults are intentionally not registered in the scheme to allow consumers to opt-out.
