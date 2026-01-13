# Package: certificates/v1alpha1

## Purpose
Provides defaulting functions and scheme registration for the certificates/v1alpha1 API version, which contains ClusterTrustBundle and PodCertificateRequest.

## Key Functions
- **addDefaultingFuncs()**: Registers defaulting functions with the scheme (calls RegisterDefaults)

## Types in v1alpha1
- **ClusterTrustBundle**: Cluster-scoped resource for distributing X.509 trust anchors
- **PodCertificateRequest**: Per-pod certificate request bound to pod lifecycle

## Design Notes
- v1alpha1 is for newer, alpha-stage certificate features
- ClusterTrustBundle enables cluster-wide CA trust distribution
- PodCertificateRequest allows pods to request certificates tied to their lifecycle
- Minimal defaulting in this version
