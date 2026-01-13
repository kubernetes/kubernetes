# Package: storage

## Purpose
Provides REST storage implementation for CSINode objects.

## Key Types

- **CSINodeStorage**: Container holding REST storage for CSINode.
- **REST**: Wraps genericregistry.Store for CSINode.

## Key Functions

- **NewStorage(optsGetter)**: Creates REST storage for CSINode:
  - Uses csinode.Strategy for all operations
  - Returns deleted object on delete
  - Includes TableConvertor for kubectl output formatting
