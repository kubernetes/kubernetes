# Package: v1beta1

## Purpose
Provides conversion and defaulting logic for discovery.k8s.io/v1beta1 API types (EndpointSlice).

## Key Functions

- **Resource(resource string)**: Returns Group-qualified GroupResource for v1beta1.
- **AddToScheme**: Registers v1beta1 types with defaulting and conversion functions.

## Key Constants

- **GroupName**: "discovery.k8s.io"
- **SchemeGroupVersion**: discovery.k8s.io/v1beta1

## Design Notes

- Beta version, maintained for backward compatibility.
- External types defined in k8s.io/api/discovery/v1beta1.
- Includes custom conversion logic for topology field migration.
