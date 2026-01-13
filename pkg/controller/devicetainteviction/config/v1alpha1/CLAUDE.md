# Package: v1alpha1

Versioned API types and defaulting for device taint eviction controller configuration.

## Key Functions

- `RecommendedDefaultDeviceTaintEvictionControllerConfiguration()`: Sets recommended defaults. Default `ConcurrentSyncs` is 8.

## Purpose

Provides the v1alpha1 versioned configuration API for the device taint eviction controller.

## Design Notes

- Default of 8 workers is a compromise between throughput and API server load
- Higher values caused issues in integration testing with pod informer watches
- Defaults are intentionally not registered in the scheme to allow consumers to opt-out
