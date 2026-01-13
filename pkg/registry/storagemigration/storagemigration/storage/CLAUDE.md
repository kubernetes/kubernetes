# Package: storage

## Purpose
Provides REST storage implementation for StorageVersionMigration objects and their status subresource.

## Key Types

- **REST**: Wraps genericregistry.Store for StorageVersionMigration main resource.
- **StatusREST**: Implements the /status subresource endpoint.

## Key Functions

- **NewREST(optsGetter)**: Creates REST storage for StorageVersionMigration:
  - Uses storagemigration.Strategy for main operations
  - Uses storagemigration.StatusStrategy for status operations
  - Returns both REST and StatusREST
  - Uses ResetFieldsStrategy for proper field isolation
  - Includes TableConvertor for kubectl output formatting

- **StatusREST.Get/Update**: Standard status subresource operations.
- **StatusREST.GetResetFields()**: Returns fields reset on status updates.
