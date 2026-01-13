# Package: install

## Purpose
Registers the imagepolicy API group with the Kubernetes API machinery, making it available for encoding/decoding operations.

## Key Functions
- `Install(scheme *runtime.Scheme)`: Registers the imagepolicy API group and its v1alpha1 version to the provided scheme, setting v1alpha1 as the preferred version.

## Key Behavior
- Auto-registers with the legacy scheme via `init()`
- Adds both internal (`imagepolicy`) and external (`v1alpha1`) types to the scheme
- Sets version priority to prefer v1alpha1

## Notes
- Uses `utilruntime.Must` to panic on registration errors (indicates programmer error)
- Follows the standard Kubernetes API group installation pattern
