# Package: storage

Provides etcd-backed REST storage for Ingress resources with status subresource support.

## Key Types

- **REST**: Main storage for Ingress CRUD operations, implements `ShortNamesProvider`.
- **StatusREST**: Dedicated endpoint for status subresource updates.

## Key Functions

- **NewREST**: Creates REST and StatusREST instances with shared underlying store.
- **ShortNames**: Returns ["ing"] for kubectl shorthand.
- **StatusREST.Get/Update**: Handles status-only operations.
- **GetResetFields**: Returns version-specific fields to reset.

## Design Notes

- Uses the standard Kubernetes generic registry pattern.
- Provides "ing" as a short name for kubectl convenience.
- Status subresource shares the store but uses separate update strategy.
- Integrates with printer subsystem for table output.
