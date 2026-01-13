# Package: v1alpha1

Versioned API types and defaulting for EndpointSlice mirroring controller configuration.

## Key Functions

- `RecommendedDefaultEndpointSliceMirroringControllerConfiguration()`: Sets recommended defaults:
  - `MirroringConcurrentServiceEndpointSyncs`: 5
  - `MirroringMaxEndpointsPerSubset`: 1000

## Key Files

- `defaults.go`: Default value functions
- `conversion.go`: Conversion functions between v1alpha1 and internal types
- `register.go`: Scheme registration

## Purpose

Provides the v1alpha1 versioned configuration API for the EndpointSlice mirroring controller.

## Design Notes

- Defaults are intentionally not registered in the scheme to allow consumers to opt-out
- 1000 max endpoints matches the legacy Endpoints capacity limit
