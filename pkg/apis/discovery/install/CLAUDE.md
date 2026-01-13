# Package: install

## Purpose
Installs the discovery API group into the Kubernetes API machinery, making EndpointSlice available for encoding/decoding.

## Key Functions

- **Install(scheme *runtime.Scheme)**: Registers discovery API group (internal, v1, v1beta1) with version priority v1 > v1beta1.
- **init()**: Auto-installs to legacy scheme on package import.

## Design Notes

- Uses utilruntime.Must() for fail-fast on registration errors.
- v1 is the preferred/stable version.
