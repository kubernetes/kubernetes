# Package: storage

Provides etcd-backed REST storage for RoleBinding resources.

## Key Types

- **REST**: Embeds `genericregistry.Store` for RoleBinding CRUD operations.

## Key Functions

- **NewREST**: Creates storage configured with RoleBinding strategy.

## Design Notes

- Basic storage layer without escalation prevention.
- Uses standard Kubernetes generic registry pattern.
- Integrates with printer subsystem for table output.
- Intended to be wrapped by policybased.Storage for production use.
