# Package: apidiscovery

## Purpose
Defines internal types for aggregated API discovery, providing efficient endpoints (/api, /apis) that return complete API resource information in a single request.

## Key Types

### APIGroupDiscoveryList
List of APIGroupDiscovery objects returned from /api and /apis endpoints. Contains aggregated API resource information from built-ins, CRDs, and aggregated servers.

### APIGroupDiscovery
Discovery information for an API group containing:
- `ObjectMeta` - Name is the API group name (can be empty for legacy v1 API)
- `Versions` - List of APIVersionDiscovery in descending preference order

### APIVersionDiscovery
Resources available in a specific API version:
- `Version` - The version name (e.g., "v1", "v1beta1")
- `Resources` - List of APIResourceDiscovery objects
- `Freshness` - Current or Stale (indicates if discovery doc is up-to-date)

### APIResourceDiscovery
Detailed resource information:
- `Resource` - Plural resource name
- `ResponseKind` - GroupVersionKind of returned objects
- `Scope` - Cluster or Namespaced
- `SingularResource` - Singular name
- `Verbs` - Supported operations
- `ShortNames`, `Categories` - CLI conveniences
- `Subresources` - List of APISubresourceDiscovery

## Constants
- `ResourceScope`: ScopeCluster, ScopeNamespace
- `DiscoveryFreshness`: DiscoveryFreshnessCurrent, DiscoveryFreshnessStale
