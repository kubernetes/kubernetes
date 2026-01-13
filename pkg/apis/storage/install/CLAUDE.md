# Package: storage/install

## Purpose
Installs the storage API group into the legacy scheme, making storage types available to the API encoding/decoding machinery.

## Key Functions
- `Install(scheme *runtime.Scheme)`: Registers all storage API versions (internal, v1, v1beta1, v1alpha1) with the given scheme and sets version priority

## Version Priority
Sets version priority order: v1 > v1beta1 > v1alpha1

## Design Notes
- Uses init() to automatically install into legacyscheme.Scheme on package import
- Registers both internal types and all versioned types (v1, v1beta1, v1alpha1)
- Follows the standard Kubernetes API group installation pattern
- Uses utilruntime.Must to panic on registration errors (should never happen in properly configured code)
