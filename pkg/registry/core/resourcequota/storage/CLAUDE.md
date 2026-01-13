# Package: storage

## Purpose
Provides REST storage implementation for ResourceQuota objects including status subresource.

## Key Types

- **REST**: Main storage for ResourceQuota operations.
- **StatusREST**: Storage for /status subresource updates.

## Key Functions

- **NewREST(optsGetter)**: Returns REST and StatusREST instances configured with:
  - ReturnDeletedObject: true
  - Uses resourcequota.MatchResourceQuota predicate
- **ShortNames()**: Returns `["quota"]` for kubectl.

## Design Notes

- Simple storage with status subresource.
- Short name "quota" for kubectl convenience.
- Returns deleted objects on delete operations.
- Status subresource uses resourcequota.StatusStrategy.
