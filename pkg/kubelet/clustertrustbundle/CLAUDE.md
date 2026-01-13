# Package clustertrustbundle

Package clustertrustbundle provides access to ClusterTrustBundle resources for projected volume support, allowing pods to mount trust anchors (CA certificates) from the cluster.

## Key Types

- `Manager`: Interface for retrieving trust anchors by name or signer
- `InformerManager[T]`: Generic informer-based implementation supporting v1alpha1 and v1beta1 APIs
- `NoopManager`: Returns errors, used in static kubelet mode
- `LazyInformerManager`: Delays initialization until first use, auto-detects available API version

## Key Functions

- `NewAlphaInformerManager`: Creates manager for v1alpha1 ClusterTrustBundle API
- `NewBetaInformerManager`: Creates manager for v1beta1 ClusterTrustBundle API
- `NewLazyInformerManager`: Creates lazy manager that auto-discovers API version
- `GetTrustAnchorsByName`: Returns PEM trust anchors from a named ClusterTrustBundle
- `GetTrustAnchorsBySigner`: Returns PEM trust anchors matching signer name and label selector

## Design Notes

- Uses LRU cache with TTL for normalized trust anchor results
- Cache invalidated on ClusterTrustBundle add/update/delete events
- Trust anchors are deduplicated and returned in randomized order (for load distribution)
- Supports allowMissing flag to return empty result instead of error
- Label selector length limited to 100KB to prevent abuse
