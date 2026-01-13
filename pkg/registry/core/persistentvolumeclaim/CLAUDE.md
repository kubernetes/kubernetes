# Package: persistentvolumeclaim

## Purpose
Provides the registry interface and REST strategy implementation for storing PersistentVolumeClaim API objects, including status subresource and data source handling per KEP 1495.

## Key Types

- **persistentvolumeclaimStrategy**: Main strategy for PVC CRUD operations (namespace-scoped).
- **persistentvolumeclaimStatusStrategy**: Strategy for /status subresource updates.

## Key Functions

- **Strategy** (var): Default logic for creating/updating PVCs.
- **StatusStrategy** (var): Strategy for status subresource.
- **NamespaceScoped()**: Returns `true` - PVCs are namespace-scoped.
- **GetResetFields()**: Returns fields to reset (status for main, spec for status strategy).
- **PrepareForCreate()**: Clears status, drops disabled fields, enforces data source backwards compatibility (KEP 1495), normalizes dataSource/dataSourceRef.
- **PrepareForUpdate()**: Preserves status, normalizes data sources for both old and new PVCs.
- **Validate()**: Validates PVC with version-aware options.
- **GetAttrs()**: Returns labels and selectable fields.
- **MatchPersistentVolumeClaim()**: Returns selection predicate for filtering.
- **PersistentVolumeClaimToSelectableFields()**: Returns filterable fields including `name` (legacy compatibility).

## Design Notes

- Namespace-scoped resource.
- Implements KEP 1495 data source handling:
  - `EnforceDataSourceBackwardsCompatibility`: Drops invalid data sources
  - `NormalizeDataSources`: Syncs dataSource and dataSourceRef fields
- Normalizes old PVC data sources during update for compatibility with older versions.
- Uses `pvcutil` for field manipulation and warnings.
