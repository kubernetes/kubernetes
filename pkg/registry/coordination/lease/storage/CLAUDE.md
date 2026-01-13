# Package: storage

## Purpose
Provides REST storage implementation for Lease resources.

## Key Types

- **REST**: Main REST storage embedding `genericregistry.Store` for Lease CRUD operations

## Key Functions

- **NewREST(optsGetter)**: Creates REST instance with configured strategies

## Design Notes

- Simple storage implementation without subresources
- No status subresource (Lease is a simple coordination primitive)
- No short names or categories defined
- Uses the lease strategy package for validation
- Leases are frequently updated (node heartbeats every ~10s)
