# Package: config

EndpointSlice controller configuration types for the kube-controller-manager.

## Key Types

- `EndpointSliceControllerConfiguration`: Contains configuration elements including:
  - `ConcurrentServiceEndpointSyncs`: Number of concurrent sync operations
  - `MaxEndpointsPerSlice`: Maximum endpoints in a single EndpointSlice
  - `EndpointUpdatesBatchPeriod`: Duration to batch endpoint updates

## Purpose

Defines the internal configuration structure used by the EndpointSlice controller. The MaxEndpointsPerSlice setting affects the tradeoff between fewer larger slices vs more smaller slices.

## Design Notes

- Part of the component-config pattern used throughout kube-controller-manager
- Larger MaxEndpointsPerSlice means fewer resources but larger individual updates
