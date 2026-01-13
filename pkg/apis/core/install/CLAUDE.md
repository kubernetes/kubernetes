# Package: install

## Purpose
Installs the core (v1) API group into the Kubernetes API machinery, making Pod, Service, Node, ConfigMap, Secret, and other core types available for encoding/decoding.

## Key Functions

- **Install(scheme *runtime.Scheme)**: Registers the core API group (internal and v1) with the given scheme and sets v1 as the preferred version.
- **init()**: Automatically installs the core API group into the legacy scheme on package import.

## Design Notes

- Uses utilruntime.Must() to panic on registration errors (fail-fast pattern).
- Only v1 version exists for core API group (no beta versions).
- This is the legacy/monolithic API and uses empty string as GroupName.
- Importing this package has the side effect of registering types.
