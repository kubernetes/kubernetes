# Package: storage

Provides etcd-backed REST storage for PodDisruptionBudget resources with status subresource support.

## Key Types

- **REST**: Main storage for PDB CRUD operations, implements `ShortNamesProvider`.
- **StatusREST**: Dedicated endpoint for status subresource updates.

## Key Functions

- **NewREST**: Creates REST and StatusREST instances with shared underlying store.
- **ShortNames**: Returns ["pdb"] for kubectl shorthand.
- **StatusREST.Get/Update**: Handles status-only operations.
- **GetResetFields**: Returns version-specific fields to reset.

## Design Notes

- Uses standard Kubernetes generic registry pattern with `genericregistry.Store`.
- Provides "pdb" as a short name for kubectl convenience.
- Status subresource shares storage but uses separate update strategy.
- Explicitly disables create-on-update for status endpoint.
