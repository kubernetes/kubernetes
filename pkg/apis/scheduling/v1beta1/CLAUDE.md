# Package: scheduling/v1beta1

## Purpose
Provides versioned (v1beta1) API types and conversion/defaulting logic for the scheduling.k8s.io API group, handling PriorityClass resources.

## Key Types/Structs
- Uses external types from `k8s.io/api/scheduling/v1beta1`
- SchemeGroupVersion: `scheduling.k8s.io/v1beta1`

## Key Functions
- `Resource(resource string)`: Returns a qualified GroupResource for the given resource name
- `AddToScheme`: Registers types with a runtime.Scheme
- `addDefaultingFuncs`: Registers defaulting functions for the v1beta1 types

## Design Notes
- This is a versioned API package following the Kubernetes API versioning pattern
- Conversion functions between internal and v1beta1 types are auto-generated (zz_generated.conversion.go)
- Defaults are auto-generated (zz_generated.defaults.go)
- The package bridges external API types with internal representations
