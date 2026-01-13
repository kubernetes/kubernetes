# Package: storage

## Purpose
Provides REST storage implementation for ResourceClaimTemplate objects.

## Key Types

- **REST**: Wraps genericregistry.Store for ResourceClaimTemplate.

## Key Functions

- **NewREST(optsGetter, nsClient)**: Creates REST storage for ResourceClaimTemplate:
  - Requires namespace client for admin access validation
  - Creates strategy via resourceclaimtemplate.NewStrategy
  - Uses PredicateFunc and AttrFunc for field selection
  - Returns error if nsClient is nil

## Design Notes

- Simpler than ResourceClaim storage since templates have no status subresource.
- Namespace client requirement ensures proper admin access validation.
