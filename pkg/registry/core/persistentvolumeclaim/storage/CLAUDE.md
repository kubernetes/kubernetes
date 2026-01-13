# Package: storage

## Purpose
Provides REST storage implementation for PersistentVolumeClaim objects including main resource and status subresource with read-time defaulting.

## Key Types

- **REST**: Main storage for PVC operations with read-time defaulting decorator.
- **StatusREST**: Storage for /status subresource updates.

## Key Functions

- **NewREST(optsGetter)**: Returns REST and StatusREST instances. Configures:
  - ReturnDeletedObject: true
  - Uses persistentvolumeclaim.MatchPersistentVolumeClaim predicate
  - Sets up Decorator for read-time defaulting

- **defaultOnRead()**: Decorator that normalizes PVCs on read operations:
  - Handles both PersistentVolumeClaim and PersistentVolumeClaimList types
  - Calls `NormalizeDataSources` to fill in missing dataSourceRef for pre-existing PVCs

- **ShortNames()**: Returns `["pvc"]` for kubectl.

## Design Notes

- Implements read-time defaulting via store Decorator pattern.
- Backfills dataSourceRef from dataSource for PVCs created before the field existed.
- Uses `pvcutil.NormalizeDataSources` for consistent data source handling.
- Status subresource uses persistentvolumeclaim.StatusStrategy.
- Shares underlying store between REST types.
