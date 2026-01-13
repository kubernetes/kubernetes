# Package: storage

Provides etcd-backed REST storage for IPAddress resources.

## Key Types

- **REST**: Embeds `genericregistry.Store`, implements `ShortNamesProvider`.

## Key Functions

- **NewREST**: Creates storage configured with IPAddress strategy.
- **ShortNames**: Returns ["ip"] for kubectl shorthand.

## Design Notes

- Simple storage with no status subresource.
- Provides "ip" as a short name for kubectl convenience.
- Uses standard Kubernetes generic registry pattern.
- Integrates with printer subsystem for table output.
