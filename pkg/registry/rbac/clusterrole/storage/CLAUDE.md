# Package: storage

Provides etcd-backed REST storage for ClusterRole resources.

## Key Types

- **REST**: Embeds `genericregistry.Store` for ClusterRole CRUD operations.

## Key Functions

- **NewREST**: Creates storage configured with ClusterRole strategy.

## Design Notes

- Basic storage layer without escalation prevention (use policybased.Storage for that).
- Uses standard Kubernetes generic registry pattern.
- Uses default table convertor for basic kubectl output.
- Intended to be wrapped by policybased.Storage for production use.
