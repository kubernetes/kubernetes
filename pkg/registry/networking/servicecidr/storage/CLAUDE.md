# Package: storage

Provides etcd-backed REST storage for ServiceCIDR resources with status subresource support.

## Key Types

- **REST**: Main storage for ServiceCIDR CRUD operations.
- **StatusREST**: Dedicated endpoint for status subresource updates.

## Key Functions

- **NewREST**: Creates REST and StatusREST instances with shared underlying store.
- **StatusREST.Get/Update**: Handles status-only operations.
- **GetResetFields**: Returns version-specific fields to reset during updates.

## Design Notes

- Uses standard Kubernetes generic registry pattern with `genericregistry.Store`.
- Status subresource shares storage but uses separate update strategy.
- Explicitly disables create-on-update for status endpoint.
- Integrates with printer subsystem for table output.
