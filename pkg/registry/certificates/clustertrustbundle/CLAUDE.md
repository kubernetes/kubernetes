# Package: clustertrustbundle

## Purpose
Implements the registry strategy for ClusterTrustBundle resources, which store trusted CA certificates for use by workloads in the cluster.

## Key Types

- **strategy**: Implements verification logic and REST strategies for ClusterTrustBundles

## Key Functions

- **Strategy**: Default logic for creating/updating/deleting ClusterTrustBundle objects
- **NamespaceScoped()**: Returns false - cluster-scoped resource
- **PrepareForCreate()**: No-op (no fields to clear)
- **PrepareForUpdate()**: No-op (no fields to preserve)
- **Validate()**: Validates using certvalidation.ValidateClusterTrustBundle
- **ValidateUpdate()**: Validates updates using certvalidation.ValidateClusterTrustBundleUpdate
- **AllowUnconditionalUpdate()**: Returns false - requires resource version

## Design Notes

- Cluster-scoped resource (no namespace)
- Simple strategy with minimal preparation logic
- No status subresource
- Part of certificates.k8s.io API group
- Feature gated by ClusterTrustBundle feature gate
