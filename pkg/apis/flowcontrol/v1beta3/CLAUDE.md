# Package: v1beta3

## Purpose
Provides conversion and defaulting logic for flowcontrol.apiserver.k8s.io/v1beta3 API types.

## Key Functions

- **Resource(resource string)**: Returns Group-qualified GroupResource for v1beta3.
- **AddToScheme**: Registers v1beta3 types with defaulting and conversion functions.

## Key Constants

- **GroupName**: "flowcontrol.apiserver.k8s.io"
- **SchemeGroupVersion**: flowcontrol.apiserver.k8s.io/v1beta3

## Design Notes

- Latest beta version before v1 stable.
- External types defined in k8s.io/api/flowcontrol/v1beta3.
- Custom conversion logic for schema evolution.
