# Package: v1beta1

## Purpose
Provides conversion and defaulting logic for coordination.k8s.io/v1beta1 API types (Lease). This package bridges the external v1beta1 API types from k8s.io/api to the internal types.

## Key Functions

- **Resource(resource string)**: Returns a Group-qualified GroupResource for v1beta1.
- **AddToScheme**: Registers v1beta1 types and conversion functions with a scheme.
- **RegisterDefaults**: Registers defaulting functions for v1beta1 types.

## Key Constants

- **GroupName**: "coordination.k8s.io"
- **SchemeGroupVersion**: coordination.k8s.io/v1beta1

## Design Notes

- Beta version of Lease API, maintained for backward compatibility.
- Uses generated conversion functions to convert between v1beta1 external and internal types.
- External types are defined in k8s.io/api/coordination/v1beta1, not in this package.
