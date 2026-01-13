# Package: install

## Purpose
Installs the apps API group into the Kubernetes API encoding/decoding machinery, registering all versions (v1, v1beta1, v1beta2) with proper version priority.

## Key Functions
- `Install(scheme *runtime.Scheme)`: Registers the internal apps types and all versioned types (v1, v1beta1, v1beta2) to the scheme, setting version priority with v1 as highest priority
- `init()`: Automatically installs to the legacy scheme on package import

## Version Priority
1. v1 (highest/preferred)
2. v1beta2
3. v1beta1

## Design Notes
- Standard Kubernetes API group installation pattern
- All three API versions are registered for backward compatibility
- v1 is the stable/preferred version for serialization
