# Package: v1alpha1

## Purpose
Provides the v1alpha1 versioned API types and registration for the imagepolicy API group, enabling image admission webhook functionality.

## Key Constants
- `GroupName`: "imagepolicy.k8s.io" - the API group name

## Key Variables
- `SchemeGroupVersion`: Identifies this as imagepolicy.k8s.io/v1alpha1
- `AddToScheme`: Function to register types with a scheme

## Key Functions
- `Resource(resource string)`: Returns a fully qualified GroupResource for the given resource name

## Code Generation
Uses Kubernetes code generators for:
- Conversion functions (`+k8s:conversion-gen`)
- Defaulting functions (`+k8s:defaulter-gen`)

## Notes
- External types are sourced from `k8s.io/api/imagepolicy/v1alpha1`
- Registers default values via `RegisterDefaults` during init
- Part of the Image Policy Webhook admission controller infrastructure
