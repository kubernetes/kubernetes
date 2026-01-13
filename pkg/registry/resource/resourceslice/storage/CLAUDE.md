# Package: storage

## Purpose
Provides REST storage implementation for ResourceSlice objects.

## Key Types

- **REST**: Wraps genericregistry.Store for ResourceSlice.

## Key Functions

- **NewREST(optsGetter)**: Creates REST storage for ResourceSlice:
  - Uses resourceslice.Strategy for all operations
  - Configures PredicateFunc and AttrFunc for field selection
  - Returns deleted object on delete
  - Includes TableConvertor for kubectl output formatting
