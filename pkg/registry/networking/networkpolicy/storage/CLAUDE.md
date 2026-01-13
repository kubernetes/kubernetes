# Package: storage

Provides etcd-backed REST storage for NetworkPolicy resources.

## Key Types

- **REST**: Embeds `genericregistry.Store`, implements `ShortNamesProvider`.

## Key Functions

- **NewREST**: Creates storage configured with NetworkPolicy strategy.
- **ShortNames**: Returns ["netpol"] for kubectl shorthand.

## Design Notes

- Simple storage with no status subresource.
- Provides "netpol" as a short name for kubectl convenience.
- Uses standard Kubernetes generic registry pattern.
- Integrates with printer subsystem for table output.
