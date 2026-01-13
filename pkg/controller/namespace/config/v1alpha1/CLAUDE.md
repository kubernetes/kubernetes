# Package: v1alpha1

Versioned API types and defaulting for Namespace controller configuration.

## Key Functions

- `RecommendedDefaultNamespaceControllerConfiguration()`: Sets recommended defaults:
  - `ConcurrentNamespaceSyncs`: 10
  - `NamespaceSyncPeriod`: 5 minutes

## Key Files

- `defaults.go`: Default value functions
- `conversion.go`: Conversion functions between v1alpha1 and internal types
- `register.go`: Scheme registration

## Purpose

Provides the v1alpha1 versioned configuration API for the Namespace controller.

## Design Notes

- Defaults are intentionally not registered in the scheme to allow consumers to opt-out
- Higher concurrency helps with faster namespace cleanup in large clusters
