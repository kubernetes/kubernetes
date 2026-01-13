# Package: storage

## Purpose
Provides REST storage implementation for PersistentVolume objects including main resource and status subresource.

## Key Types

- **REST**: Main storage for PersistentVolume operations.
- **StatusREST**: Storage for /status subresource updates.

## Key Functions

- **NewREST(optsGetter)**: Returns REST and StatusREST instances. Configures:
  - ReturnDeletedObject: true
  - Uses persistentvolume.MatchPersistentVolumes predicate
  - Separate stores with different strategies for status

- **ShortNames()**: Returns `["pv"]` for kubectl.

## Design Notes

- Implements `rest.ShortNamesProvider` interface.
- Returns deleted object on delete operations.
- Status subresource uses persistentvolume.StatusStrategy.
- Shares underlying store between REST types.
- Implements `rest.ResetFieldsStrategy` for server-side apply.
