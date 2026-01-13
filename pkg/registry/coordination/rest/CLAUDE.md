# Package: rest

## Purpose
Provides the REST storage provider for the "coordination" API group, wiring up Lease and LeaseCandidate resources to the API server.

## Key Types

- **RESTStorageProvider**: Implements the storage provider interface for coordination API group

## Key Functions

- **NewRESTStorage(apiResourceConfigSource, restOptionsGetter)**: Creates APIGroupInfo with storage handlers
- **v1Storage()**: Creates storage for coordination.k8s.io/v1:
  - leases
- **v1beta1Storage()**: Creates storage for coordination.k8s.io/v1beta1:
  - leasecandidates
- **v1alpha2Storage()**: Creates storage for coordination.k8s.io/v1alpha2:
  - leasecandidates
- **GroupName()**: Returns "coordination.k8s.io"

## Design Notes

- Leases are in v1 (stable)
- LeaseCandidates are available in both v1beta1 and v1alpha2
- LeaseCandidates are used for coordinated leader election
- Note: When adding versions, also update aggregator.go priorities
