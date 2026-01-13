# Package: storage

## Purpose
Provides REST storage implementation for PriorityClass objects.

## Key Types

- **REST**: Wraps genericregistry.Store for PriorityClass.

## Key Functions

- **NewREST(optsGetter)**: Creates REST storage for PriorityClass:
  - Uses priorityclass.Strategy for all operations
  - Returns deleted object on delete
  - Includes TableConvertor for kubectl output formatting

- **ShortNames()**: Returns ["pc"] - the short name for PriorityClass.
