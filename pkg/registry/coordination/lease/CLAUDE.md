# Package: lease

## Purpose
Implements the registry strategy for Lease resources, which are used for leader election and node heartbeats in Kubernetes.

## Key Types

- **leaseStrategy**: Implements verification logic and REST strategies for Leases

## Key Functions

- **Strategy**: Default logic for creating/updating Lease objects
- **NamespaceScoped()**: Returns true - Leases are namespace-scoped
- **PrepareForCreate()**: Drops Strategy and PreferredHolder fields if CoordinatedLeaderElection feature is disabled
- **PrepareForUpdate()**: Preserves Strategy/PreferredHolder from old object if feature disabled and not previously set
- **Validate()**: Validates new Leases using validation.ValidateLease
- **ValidateUpdate()**: Validates Lease updates
- **AllowCreateOnUpdate()**: Returns true - can create with PUT request
- **AllowUnconditionalUpdate()**: Returns false - requires resource version

## Design Notes

- Namespace-scoped resource
- Used for node heartbeats (kube-node-lease namespace) and leader election
- Supports CoordinatedLeaderElection feature gate for Strategy and PreferredHolder fields
- AllowCreateOnUpdate=true enables upsert semantics (important for node heartbeats)
- No status subresource (Lease spec contains all mutable fields)
