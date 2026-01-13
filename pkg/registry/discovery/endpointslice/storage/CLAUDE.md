# Package: storage

Provides etcd-backed REST storage implementation for EndpointSlice resources.

## Key Types

- **REST**: Embeds `genericregistry.Store` to provide RESTful CRUD operations for EndpointSlices against etcd.

## Key Functions

- **NewREST**: Creates and configures a new REST storage instance with:
  - Object factories for EndpointSlice and EndpointSliceList
  - Create/Update/Delete strategies from the parent endpointslice package
  - Table conversion for kubectl output formatting

## Design Notes

- Uses the standard Kubernetes generic registry pattern.
- Delegates all strategy logic (validation, preparation) to `endpointslice.Strategy`.
- Integrates with the printer subsystem for human-readable table output.
