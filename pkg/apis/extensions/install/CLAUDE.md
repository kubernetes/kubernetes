# Package: install

## Purpose
Installs the extensions API group into the Kubernetes API machinery.

## Key Functions

- **Install(scheme *runtime.Scheme)**: Registers extensions API group (internal and v1beta1) with v1beta1 as preferred version.
- **init()**: Auto-installs to legacy scheme on package import.

## Design Notes

- Only v1beta1 version exists (no stable version for extensions group).
- Uses utilruntime.Must() for fail-fast on registration errors.
- Extensions group is deprecated; types have moved to dedicated API groups.
