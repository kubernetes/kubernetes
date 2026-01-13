# Package: install

## Purpose
Installs the admission API group into the legacy scheme, making it available for encoding/decoding admission review objects.

## Key Functions
- `Install(scheme *runtime.Scheme)` - Registers all admission types and versions with the given scheme
- `init()` - Automatically installs to the legacy scheme on package import

## Registered Versions
1. Internal version (hub) - `k8s.io/kubernetes/pkg/apis/admission`
2. v1beta1 - `k8s.io/api/admission/v1beta1`
3. v1 - `k8s.io/api/admission/v1`

## Version Priority
v1 > v1beta1 (v1 is preferred)

## Design Notes
- Uses the standard install pattern for Kubernetes API groups
- Panics on registration errors via `utilruntime.Must()`
