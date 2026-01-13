# Package: storagemigration/install

## Purpose
Installs the storagemigration API group into the legacy scheme, making storage version migration types available to the API encoding/decoding machinery.

## Key Functions
- `Install(scheme *runtime.Scheme)`: Registers storagemigration API versions (internal, v1beta1) with the given scheme

## Design Notes
- Uses init() to automatically install into legacyscheme.Scheme on package import
- Registers both internal types and versioned types (v1beta1)
- Follows the standard Kubernetes API group installation pattern
- Uses utilruntime.Must to panic on registration errors
