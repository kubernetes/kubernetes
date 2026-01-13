# Package: storageversion

This package provides the registry strategy for StorageVersion resources, which track the encoding version used for each API resource in etcd.

## Key Types

- `storageVersionStrategy` - Main strategy for spec updates
- `storageVersionStatusStrategy` - Strategy for status subresource updates

## Key Functions

- `Strategy` - Package-level singleton for spec operations
- `StatusStrategy` - Package-level singleton for status operations
- `PrepareForCreate()` - Clears status on creation
- `PrepareForUpdate()` - Preserves status on spec update
- `Validate()` - Validates StorageVersion spec
- `GetResetFields()` - Returns fields reset by each strategy

## Design Notes

- Cluster-scoped resource
- Separates spec and status update paths
- Status tracks which API servers are using which versions
- Spec is immutable after creation (name matches resource)
- Used for storage migration coordination
- Helps ensure all API servers agree on encoding version
