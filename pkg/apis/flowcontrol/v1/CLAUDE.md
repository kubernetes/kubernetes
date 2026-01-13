# Package: v1

## Purpose
Provides conversion and defaulting logic for flowcontrol.apiserver.k8s.io/v1 API types.

## Key Functions

- **Resource(resource string)**: Returns Group-qualified GroupResource for v1.
- **AddToScheme**: Registers v1 types with defaulting functions.

## Key Constants

- **GroupName**: "flowcontrol.apiserver.k8s.io"
- **SchemeGroupVersion**: flowcontrol.apiserver.k8s.io/v1

## Design Notes

- Stable version of the flowcontrol API.
- External types defined in k8s.io/api/flowcontrol/v1.
- Generated conversion and defaults in zz_generated files.
