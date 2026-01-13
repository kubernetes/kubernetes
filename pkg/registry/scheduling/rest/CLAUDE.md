# Package: rest

## Purpose
Provides the RESTStorageProvider for the scheduling.k8s.io API group, which wires up scheduling-related resources to the API server.

## Key Types

- **RESTStorageProvider**: Implements genericapiserver.RESTStorageProvider for the scheduling API group.

## Key Functions

- **NewRESTStorage(apiResourceConfigSource, restOptionsGetter)**: Creates APIGroupInfo with storage for scheduling API versions.
- **v1Storage(...)**: Creates storage map for v1 resources (PriorityClass).
- **v1alpha2Storage(...)**: Creates storage map for v1alpha2 resources (Workload, PodGroup).
- **GroupName()**: Returns "scheduling.k8s.io".

## Registered Resources

- **v1**: priorityclasses
- **v1alpha2**: workloads, workloads/status, podgroups, podgroups/status (feature-gated)

## Feature Gating

- Workload and PodGroup resources require the GenericWorkload feature gate to be enabled.
