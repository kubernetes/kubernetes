# Package: v1alpha1

Versioned API types and defaulting for Endpoint controller configuration.

## Key Functions

- `RecommendedDefaultEndpointControllerConfiguration()`: Sets recommended defaults. Default `ConcurrentEndpointSyncs` is 5.

## Key Files

- `defaults.go`: Default value functions
- `conversion.go`: Conversion functions between v1alpha1 and internal types
- `register.go`: Scheme registration

## Purpose

Provides the v1alpha1 versioned configuration API for the Endpoint controller.

## Design Notes

- Defaults are intentionally not registered in the scheme to allow consumers to opt-out
- EndpointUpdatesBatchPeriod defaults to 0 (no batching)
