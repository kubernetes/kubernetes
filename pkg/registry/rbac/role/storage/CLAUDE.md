# Package: storage

Provides etcd-backed REST storage for Role resources.

## Key Types

- **REST**: Embeds `genericregistry.Store` for Role CRUD operations.

## Key Functions

- **NewREST**: Creates storage configured with Role strategy.

## Design Notes

- Basic storage layer without escalation prevention.
- Uses standard Kubernetes generic registry pattern.
- Uses default table convertor.
- Intended to be wrapped by policybased.Storage for production use.
