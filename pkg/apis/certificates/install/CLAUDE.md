# Package: certificates/install

## Purpose
Registers the certificates API group with all its versions into a Kubernetes API scheme, making CSR and trust bundle resources available to the API server.

## Key Functions
- **Install(scheme)**: Registers the certificates API group and all versions with the provided scheme
- **init()**: Automatically installs to the legacy scheme on package import

## Registered Versions
- `certificates.k8s.io/v1` (preferred/storage version)
- `certificates.k8s.io/v1beta1` (deprecated)
- `certificates.k8s.io/v1alpha1` (for ClusterTrustBundle)

## Version Priority
v1 is the preferred version and is listed first in the installation order.

## Design Notes
- Uses `utilruntime.Must()` to panic on registration errors (fail-fast)
- Follows the standard Kubernetes API group installation pattern
- Auto-registers via init() for legacy scheme compatibility
- v1alpha1 contains newer features like ClusterTrustBundle
