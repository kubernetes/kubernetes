# Package: v1beta1

## Purpose
Provides conversion and defaulting logic for flowcontrol.apiserver.k8s.io/v1beta1 API types.

## Key Functions

- **Resource(resource string)**: Returns Group-qualified GroupResource for v1beta1.
- **AddToScheme**: Registers v1beta1 types with defaulting and conversion functions.

## Key Constants

- **GroupName**: "flowcontrol.apiserver.k8s.io"
- **SchemeGroupVersion**: flowcontrol.apiserver.k8s.io/v1beta1

## Design Notes

- Early beta version, maintained for backward compatibility.
- External types defined in k8s.io/api/flowcontrol/v1beta1.
- Custom conversion logic for schema evolution.
