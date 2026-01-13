# Package: storage

## Purpose
Provides REST storage implementation for StorageClass objects.

## Key Types

- **REST**: Wraps genericregistry.Store for StorageClass.

## Key Functions

- **NewREST(optsGetter)**: Creates REST storage for StorageClass:
  - Uses storageclass.Strategy for all operations
  - Returns deleted object on delete
  - Includes TableConvertor for kubectl output formatting

- **ShortNames()**: Returns ["sc"] - the short name for StorageClass.
