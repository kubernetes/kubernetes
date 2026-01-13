# Package: persistentvolume

## Purpose
Provides the registry interface and REST strategy implementation for storing PersistentVolume API objects, including status subresource handling.

## Key Types

- **persistentvolumeStrategy**: Main strategy for PersistentVolume CRUD operations (cluster-scoped).
- **persistentvolumeStatusStrategy**: Strategy for /status subresource updates.

## Key Functions

- **Strategy** (var): Default logic for creating/updating PersistentVolumes.
- **StatusStrategy** (var): Strategy for status subresource.
- **NamespaceScoped()**: Returns `false` - PVs are cluster-scoped.
- **GetResetFields()**: Returns fields to reset (status for main, spec for status strategy).
- **PrepareForCreate()**: Clears status, drops disabled spec fields, sets phase to Pending with LastPhaseTransitionTime.
- **PrepareForUpdate()**: Preserves status from old object, drops disabled spec fields.
- **Validate()**: Validates PV using both core validation and volume-specific validation.
- **PrepareForUpdate (status)**: Preserves spec, auto-updates LastPhaseTransitionTime when phase changes.
- **GetAttrs()**: Returns labels and selectable fields.
- **MatchPersistentVolumes()**: Returns selection predicate for filtering.
- **PersistentVolumeToSelectableFields()**: Returns filterable fields including `name` (legacy compatibility).

## Design Notes

- Cluster-scoped resource.
- Uses `NowFunc` variable for testability of timestamp handling.
- Auto-manages `LastPhaseTransitionTime` on phase changes.
- Dual validation: core validation + volume-specific validation.
- `pvutil.DropDisabledSpecFields` handles feature-gated fields.
