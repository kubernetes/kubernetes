# Package: storage

Provides etcd-backed REST storage for RuntimeClass resources.

## Key Types

- **REST**: Embeds `genericregistry.Store` for RuntimeClass CRUD operations.

## Key Functions

- **NewREST**: Creates storage configured with RuntimeClass strategy.
- **ObjectNameFunc**: Custom function to extract name from RuntimeClass objects.

## Design Notes

- Simple storage with no status subresource.
- Uses standard Kubernetes generic registry pattern.
- Includes explicit ObjectNameFunc for name extraction.
- Integrates with printer subsystem for table output.
