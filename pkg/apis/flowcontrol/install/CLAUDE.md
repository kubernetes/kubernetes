# Package: install

## Purpose
Installs the flowcontrol API group into the Kubernetes API machinery.

## Key Functions

- **Install(scheme *runtime.Scheme)**: Registers flowcontrol API group (internal, v1, v1beta3, v1beta2, v1beta1) with version priority v1 > v1beta3 > v1beta2 > v1beta1.
- **init()**: Auto-installs to legacy scheme on package import.

## Design Notes

- Uses utilruntime.Must() for fail-fast on registration errors.
- v1 is the stable/preferred version.
- Multiple beta versions maintained for backward compatibility.
