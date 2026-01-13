# Package: storage

## Purpose
Provides REST storage implementation for CSIDriver objects.

## Key Types

- **CSIDriverStorage**: Container holding REST storage for CSIDriver.
- **REST**: Wraps genericregistry.Store for CSIDriver.

## Key Functions

- **NewStorage(optsGetter)**: Creates REST storage for CSIDriver:
  - Uses csidriver.Strategy for all operations
  - Returns deleted object on delete
  - Includes TableConvertor for kubectl output formatting
