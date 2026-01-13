# Package podresources

Package podresources implements the Pod Resources API server and client for exposing resource allocation information (CPUs, devices, memory, dynamic resources) to monitoring agents.

## Key Types

- `DevicesProvider`: Interface for querying device allocations per container
- `PodsProvider`: Interface for accessing pod information on the node
- `CPUsProvider`: Interface for querying CPU allocations per container
- `MemoryProvider`: Interface for querying memory allocations per container
- `DynamicResourcesProvider`: Interface for querying DRA (Dynamic Resource Allocation) resources
- `PodResourcesProviders`: Aggregates all provider interfaces
- `v1PodResourcesServer`: Implements the PodResourcesLister gRPC service

## Key Functions

- `NewV1PodResourcesServer`: Creates the v1 PodResources gRPC server
- `GetV1Client`: Creates a client connection to the PodResources v1 API
- `GetV1alpha1Client`: Creates a client connection to the deprecated v1alpha1 API

## Server Methods

- `List`: Returns resources assigned to all non-terminal pods
- `Get`: Returns resources for a specific pod (requires KubeletPodResourcesGet feature gate)
- `GetAllocatableResources`: Returns total allocatable devices, CPUs, and memory

## Design Notes

- Serves via Unix domain socket for local access
- Supports both v1 and v1alpha1 (deprecated) API versions
- Filters out terminal pods (Failed/Succeeded) from resource listings
- Feature-gated support for dynamic resources and Get method
