# Package: config

EndpointSlice mirroring controller configuration types for the kube-controller-manager.

## Key Types

- `EndpointSliceMirroringControllerConfiguration`: Contains configuration elements including:
  - `MirroringConcurrentServiceEndpointSyncs`: Number of concurrent sync operations
  - `MirroringMaxEndpointsPerSubset`: Maximum endpoints to mirror per EndpointSubset
  - `MirroringEndpointUpdatesBatchPeriod`: Duration to batch updates

## Purpose

Defines the internal configuration structure used by the EndpointSlice mirroring controller.

## Design Notes

- Part of the component-config pattern used throughout kube-controller-manager
- MaxEndpointsPerSubset limits how many addresses are mirrored from a single Endpoints subset
