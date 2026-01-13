# Package: storage

Provides etcd-backed REST storage for IngressClass resources.

## Key Types

- **REST**: Embeds `genericregistry.Store` for IngressClass CRUD operations.

## Key Functions

- **NewREST**: Creates and configures storage with IngressClass strategy.

## Design Notes

- Simple storage with no status subresource (unlike Ingress).
- Uses standard Kubernetes generic registry pattern.
- Integrates with printer subsystem for table output.
