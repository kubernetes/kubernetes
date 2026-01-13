# Package: v1beta1

## Purpose
Provides the v1beta1 versioned API registration and defaults for the rbac.authorization.k8s.io API group.

## Key Constants
- `GroupName`: "rbac.authorization.k8s.io"
- `SchemeGroupVersion`: rbac.authorization.k8s.io/v1beta1

## Key Functions
- `Resource(resource string)`: Returns qualified GroupResource
- `AddToScheme`: Registers types with a scheme
- `addDefaultingFuncs`: Registers defaulting functions

## Code Generation Markers
- `+k8s:conversion-gen`: Generates conversion to/from internal rbac types
- `+k8s:defaulter-gen`: Generates defaulting functions
- `+k8s:validation-gen`: Generates validation functions

## Notes
- External types sourced from `k8s.io/api/rbac/v1beta1`
- Beta version; structure matches v1
- Contains helper functions similar to v1 for building RBAC objects
