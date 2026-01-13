# Package: resolver

This package provides a resource resolver that maps GroupVersionKind (GVK) to GroupVersionResource (GVR) using the discovery API.

## Key Types

- `ResourceResolver` - Interface for resolving GVK to GVR
- `discoveryResourceResolver` - Implementation using discovery client
- `ResourceResolverFunc` - Function adapter for the interface

## Key Functions

- `NewDiscoveryResourceResolver()` - Creates a resolver using the discovery client
- `Resolve()` - Converts a GVK to a GVR by querying server resources

## Design Notes

- Used by admission policy validation to resolve paramKind references
- Queries the API server's discovery endpoint for resource mappings
- Ignores subresources (resources with "/" in name)
- Returns NoKindMatchError if the kind is not found
- Enables validation that paramKind references exist on the server
