# Package: v1

## Purpose
Provides conversion and defaulting logic for discovery.k8s.io/v1 API types (EndpointSlice).

## Key Functions

- **Resource(resource string)**: Returns Group-qualified GroupResource for v1.
- **AddToScheme**: Registers v1 types with defaulting and conversion functions.

## Key Constants

- **GroupName**: "discovery.k8s.io"
- **SchemeGroupVersion**: discovery.k8s.io/v1

## Design Notes

- External types defined in k8s.io/api/discovery/v1.
- Generated conversion in zz_generated.conversion.go.
- Generated defaults in zz_generated.defaults.go.
- Generated validation in zz_generated.validations.go.
