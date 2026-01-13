# Package: coordination

## Purpose
Internal (unversioned) API types for the coordination.k8s.io API group, providing distributed coordination primitives for Kubernetes components, primarily used for leader election.

## Key Types

- **Lease**: Represents a lease for distributed coordination. Contains holder identity, duration, acquire/renew times, and transition count. Supports coordinated leader election via Strategy field.
- **LeaseSpec**: Specification including HolderIdentity, LeaseDurationSeconds, AcquireTime, RenewTime, LeaseTransitions, Strategy, and PreferredHolder.
- **LeaseCandidate**: Defines a candidate for coordinated leader election with version information for intelligent leader selection.
- **LeaseCandidateSpec**: Contains LeaseName, PingTime, RenewTime, BinaryVersion, EmulationVersion, and Strategy.
- **CoordinatedLeaseStrategy**: String type for leader election strategies (e.g., OldestEmulationVersion).

## Key Functions

- **Kind(kind string)**: Returns a Group-qualified GroupKind for the given kind.
- **Resource(resource string)**: Returns a Group-qualified GroupResource.
- **AddToScheme**: Registers types with a runtime.Scheme.
- **addKnownTypes(scheme)**: Registers Lease, LeaseList, LeaseCandidate, LeaseCandidateList.

## Design Notes

- This is the internal version; external versioned types are in v1, v1alpha2, v1beta1 subdirectories.
- CoordinatedLeaderElection feature gate enables Strategy and PreferredHolder fields.
- OldestEmulationVersion strategy picks leaders based on emulation version, then binary version, then creation timestamp.
