# Package: persistentvolumeclaim

## Purpose
Provides utilities for PersistentVolumeClaim API objects including feature-gated field handling, data source normalization, and deprecation warnings.

## Key Functions

### Field Management
- `DropDisabledFields(pvcSpec, oldPVCSpec *core.PersistentVolumeClaimSpec)` - Removes feature-gated fields when disabled:
  - VolumeAttributesClassName (VolumeAttributesClass feature)
  - DataSourceRef (AnyVolumeDataSource feature)
  - Cross-namespace DataSourceRef (CrossNamespaceVolumeDataSource feature)
- `DropDisabledFieldsFromStatus(pvc, oldPVC *core.PersistentVolumeClaim)` - Handles status fields for VolumeAttributesClass and RecoverVolumeExpansionFailure features
- `EnforceDataSourceBackwardsCompatibility(pvcSpec, oldPVCSpec *core.PersistentVolumeClaimSpec)` - Drops invalid data sources for backwards compatibility (KEP 1495)
- `NormalizeDataSources(pvcSpec *core.PersistentVolumeClaimSpec)` - Ensures DataSource and DataSourceRef have consistent contents

### Warnings
- `GetWarningsForPersistentVolumeClaim(pv *core.PersistentVolumeClaim) []string` - Warns about deprecated storage class annotation
- `GetWarningsForPersistentVolumeClaimSpec(fieldPath *field.Path, pvSpec core.PersistentVolumeClaimSpec) []string` - Warns about fractional byte values in storage requests/limits

## Design Notes
- Supports both legacy DataSource field and newer DataSourceRef field for volume cloning/snapshots
- Only PersistentVolumeClaim and VolumeSnapshot are valid DataSource types in the legacy field
