# Package: storage

## Purpose
Provides REST storage implementation for Secret objects.

## Key Types

- **REST**: Main storage for Secret operations using genericregistry.Store.

## Key Functions

- **NewREST(optsGetter)**: Returns REST instance configured with:
  - ReturnDeletedObject: true
  - Uses secret.Matcher predicate with type field indexing

## Design Notes

- Simple storage implementation with no subresources.
- Returns deleted objects on delete operations.
- Leverages field indexing on `type` from strategy for efficient queries.
- Uses standard genericregistry.Store pattern.
