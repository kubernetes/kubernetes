# Package: install

## Purpose
Installs the coordination API group into the Kubernetes API machinery, making Lease and LeaseCandidate types available for encoding/decoding across all supported versions.

## Key Functions

- **Install(scheme *runtime.Scheme)**: Registers the coordination API group (internal and versioned types) with the given scheme and sets version priority (v1 > v1beta1 > v1alpha2).
- **init()**: Automatically installs the coordination API group into the legacy scheme on package import.

## Design Notes

- Uses utilruntime.Must() to panic on registration errors (fail-fast pattern).
- Version priority ensures v1 is preferred for serialization when multiple versions are available.
- Imports and registers all coordination API versions: internal, v1, v1alpha2, v1beta1.
