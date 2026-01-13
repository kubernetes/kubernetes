# Package: v1alpha1

Versioned API types and defaulting for DaemonSet controller configuration.

## Key Functions

- `RecommendedDefaultDaemonSetControllerConfiguration()`: Sets recommended defaults for DaemonSetControllerConfiguration. Default `ConcurrentDaemonSetSyncs` is 2.

## Key Files

- `defaults.go`: Default value functions
- `conversion.go`: Conversion functions between v1alpha1 and internal types
- `register.go`: Scheme registration

## Purpose

Provides the v1alpha1 versioned configuration API for the DaemonSet controller. This follows the Kubernetes component-config pattern where configuration types are versioned APIs that can evolve independently.

## Design Notes

- Defaults are intentionally not registered in the scheme to allow consumers to opt-out
- Consumers should call the defaulting function in their own `SetDefaults_` methods
