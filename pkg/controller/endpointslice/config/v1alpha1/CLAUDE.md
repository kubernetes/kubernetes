# Package: v1alpha1

Versioned API types and defaulting for EndpointSlice controller configuration.

## Key Functions

- `RecommendedDefaultEndpointSliceControllerConfiguration()`: Sets recommended defaults:
  - `ConcurrentServiceEndpointSyncs`: 5
  - `MaxEndpointsPerSlice`: 100

## Key Files

- `defaults.go`: Default value functions
- `conversion.go`: Conversion functions between v1alpha1 and internal types
- `register.go`: Scheme registration

## Purpose

Provides the v1alpha1 versioned configuration API for the EndpointSlice controller.

## Design Notes

- Defaults are intentionally not registered in the scheme to allow consumers to opt-out
- 100 endpoints per slice is a balance between resource count and update size
