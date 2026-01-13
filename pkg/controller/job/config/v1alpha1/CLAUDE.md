# Package: v1alpha1

Versioned API types and defaulting for Job controller configuration.

## Key Functions

- `RecommendedDefaultJobControllerConfiguration()`: Sets recommended defaults. Default `ConcurrentJobSyncs` is 5.

## Key Files

- `defaults.go`: Default value functions
- `conversion.go`: Conversion functions between v1alpha1 and internal types
- `register.go`: Scheme registration

## Purpose

Provides the v1alpha1 versioned configuration API for the Job controller.

## Design Notes

- Defaults are intentionally not registered in the scheme to allow consumers to opt-out
