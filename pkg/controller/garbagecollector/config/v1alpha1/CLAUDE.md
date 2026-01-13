# Package: v1alpha1

Versioned API types and defaulting for garbage collector controller configuration.

## Key Functions

- `RecommendedDefaultGarbageCollectorControllerConfiguration()`: Sets recommended defaults:
  - `EnableGarbageCollector`: true
  - `ConcurrentGCSyncs`: 20

## Key Files

- `defaults.go`: Default value functions
- `conversion.go`: Conversion functions between v1alpha1 and internal types
- `register.go`: Scheme registration

## Purpose

Provides the v1alpha1 versioned configuration API for the garbage collector controller.

## Design Notes

- Defaults are intentionally not registered in the scheme to allow consumers to opt-out
- GC is enabled by default; disabling requires explicit configuration
