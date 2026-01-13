# Package: storage

Provides etcd-backed REST storage for PriorityLevelConfiguration resources with status subresource.

## Key Types

- **PriorityLevelConfigurationStorage**: Container for REST and StatusREST instances.
- **REST**: Main storage for PriorityLevelConfiguration CRUD operations.
- **StatusREST**: Dedicated endpoint for status subresource updates.

## Key Functions

- **NewREST**: Creates both REST and StatusREST instances with shared underlying store.
- **StatusREST.Get/Update**: Handles status-only operations.
- **GetResetFields**: Returns version-specific fields to reset during updates.

## Design Notes

- Follows the standard Kubernetes storage pattern with `genericregistry.Store`.
- Status subresource explicitly disables create-on-update behavior.
- Both endpoints share storage but use different strategies for proper field isolation.
- Integrates with the printer system for kubectl table output.
