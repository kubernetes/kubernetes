# Package: v1

## Purpose
Provides conversion and defaulting logic for coordination.k8s.io/v1 API types (Lease). This package bridges the external v1 API types from k8s.io/api to the internal types.

## Key Functions

- **Resource(resource string)**: Returns a Group-qualified GroupResource for v1.
- **AddToScheme**: Registers v1 types and conversion functions with a scheme.
- **RegisterDefaults**: Registers defaulting functions for v1 types.

## Key Constants

- **GroupName**: "coordination.k8s.io"
- **SchemeGroupVersion**: coordination.k8s.io/v1

## Design Notes

- Uses generated conversion functions (zz_generated.conversion.go) to convert between v1 external and internal types.
- Uses generated defaulting functions (zz_generated.defaults.go) to set default values.
- External types are defined in k8s.io/api/coordination/v1, not in this package.
