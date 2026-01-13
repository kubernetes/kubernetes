# Package: storage

## Purpose
Provides REST storage implementation for ResourceClaim objects and their status subresource.

## Key Types

- **REST**: Wraps genericregistry.Store for ResourceClaim main resource.
- **StatusREST**: Implements the /status subresource endpoint.

## Key Functions

- **NewREST(optsGetter, nsClient)**: Creates REST storage for ResourceClaim:
  - Requires namespace client for admin access validation
  - Creates strategy via resourceclaim.NewStrategy
  - Uses PredicateFunc and AttrFunc for field selection
  - Returns both REST and StatusREST

- **StatusREST.Get/Update**: Standard status subresource operations.
- **StatusREST.GetResetFields()**: Returns fields reset on status updates.

## Design Notes

- Namespace client is required - returns error if nil.
- Status store shares underlying storage with main store but uses separate strategy.
