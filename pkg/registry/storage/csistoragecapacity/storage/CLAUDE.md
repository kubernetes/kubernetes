# Package: storage

## Purpose
Provides REST storage implementation for CSIStorageCapacity objects.

## Key Types

- **CSIStorageCapacityStorage**: Container holding REST storage for CSIStorageCapacity.
- **REST**: Wraps genericregistry.Store for CSIStorageCapacity.

## Key Functions

- **NewStorage(optsGetter)**: Creates REST storage for CSIStorageCapacity:
  - Uses csistoragecapacity.Strategy for all operations
  - Uses default TableConvertor (no custom printer handlers)
