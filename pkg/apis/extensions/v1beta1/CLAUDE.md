# Package: v1beta1

## Purpose
Provides conversion and defaulting logic for extensions/v1beta1 API types.

## Key Functions

- **Resource(resource string)**: Returns Group-qualified GroupResource for v1beta1.
- **AddToScheme**: Registers v1beta1 types with defaulting functions.
- **addDefaultingFuncs**: Registers defaulting logic.

## Key Constants

- **GroupName**: "extensions"
- **SchemeGroupVersion**: extensions/v1beta1

## Design Notes

- External types defined in k8s.io/api/extensions/v1beta1.
- Deprecated group; use apps/v1, networking.k8s.io/v1 instead.
- Conversion logic handles mapping to/from canonical API groups.
