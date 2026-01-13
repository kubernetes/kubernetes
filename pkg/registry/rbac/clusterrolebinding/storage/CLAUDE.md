# Package: storage

Provides etcd-backed REST storage for ClusterRoleBinding resources.

## Key Types

- **REST**: Embeds `genericregistry.Store` for ClusterRoleBinding CRUD operations.

## Key Functions

- **NewREST**: Creates storage configured with ClusterRoleBinding strategy.

## Design Notes

- Basic storage layer without escalation prevention.
- Uses standard Kubernetes generic registry pattern.
- Integrates with printer subsystem for table output.
- Intended to be wrapped by policybased.Storage for production use.
