# Package: storage

Provides etcd-backed REST storage for FlowSchema resources with status subresource support.

## Key Types

- **FlowSchemaStorage**: Container holding both main REST and StatusREST instances.
- **REST**: Main storage for FlowSchema CRUD operations.
- **StatusREST**: Dedicated storage endpoint for status subresource updates.

## Key Functions

- **NewREST**: Creates both REST and StatusREST instances sharing the same underlying store.
- **StatusREST.Get/Update**: Handles status-only reads and updates.
- **GetResetFields**: Returns fields that should be reset during updates (delegates to store).

## Design Notes

- Uses the standard Kubernetes generic registry pattern with `genericregistry.Store`.
- Status subresource uses a separate strategy that preserves spec during status updates.
- Both REST endpoints share the underlying store but use different strategies.
- Integrates with printer subsystem for table output.
