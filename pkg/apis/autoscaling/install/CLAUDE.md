# Package: install

## Purpose
Installs the autoscaling API group into the Kubernetes API encoding/decoding machinery, registering all versions (v1, v2) with proper version priority.

## Key Functions
- `Install(scheme *runtime.Scheme)`: Registers the internal autoscaling types and all versioned types to the scheme
- `init()`: Automatically installs to the legacy scheme on package import

## Version Priority
1. v2 (highest/preferred)
2. v1

## Design Notes
- Standard Kubernetes API group installation pattern
- v2 is the preferred version with full HPA behavior support
- v1 is maintained for backward compatibility (limited metrics support)
