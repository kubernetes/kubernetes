# Package: storage

## Purpose
Provides REST storage implementation for Carp objects and their status subresource.

## Key Types

- **REST**: Wraps genericregistry.Store for Carp main resource.
- **StatusREST**: Implements the /status subresource endpoint.

## Key Functions

- **NewREST(optsGetter, nsClient)**: Creates REST storage for Carp:
  - Requires namespace client
  - Creates strategy via carp.NewStrategy
  - Uses PredicateFunc and AttrFunc for field selection
  - Returns both REST and StatusREST
  - Returns deleted object on delete
  - Includes TableConvertor for kubectl output formatting

- **StatusREST.Get/Update**: Standard status subresource operations.
- **StatusREST.GetResetFields()**: Returns fields reset on status updates.

## Design Notes

- Part of the test API group for validating registry patterns.
- Namespace client is required - returns error if nil.
