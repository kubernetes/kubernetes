# Package: storage

## Purpose
Provides REST storage implementation for Workload objects and their status subresource.

## Key Types

- **REST**: Wraps genericregistry.Store for Workload main resource.
- **StatusREST**: Implements the /status subresource endpoint.

## Key Functions

- **NewREST(optsGetter)**: Creates REST storage for Workload:
  - Uses workload.Strategy for main operations
  - Returns both REST and StatusREST
  - Configures PredicateFunc and AttrFunc for field selection
  - Includes TableConvertor for kubectl output formatting

- **StatusREST.Get/Update**: Standard status subresource operations.
- **StatusREST.GetResetFields()**: Returns fields reset on status updates.

## Design Notes

- Part of the GenericWorkload feature gate.
- Status store shares underlying storage with main store.
