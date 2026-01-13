# Package: rest

## Purpose
Provides the REST storage provider for the "certificates" API group, wiring up CSR, ClusterTrustBundle, and PodCertificateRequest resources to the API server.

## Key Types

- **RESTStorageProvider**: Implements the storage provider interface
  - Requires Authorizer for PodCertificateRequest status updates

## Key Functions

- **NewRESTStorage(apiResourceConfigSource, restOptionsGetter)**: Creates APIGroupInfo with storage handlers
- **v1Storage()**: Creates storage for certificates/v1:
  - certificatesigningrequests, /status, /approval
- **v1beta1Storage()**: Creates storage for certificates/v1beta1:
  - clustertrustbundles (if ClusterTrustBundle feature enabled)
  - podcertificaterequests, /status (if PodCertificateRequest feature enabled)
- **v1alpha1Storage()**: Creates storage for certificates/v1alpha1:
  - clustertrustbundles (if ClusterTrustBundle feature enabled)
- **GroupName()**: Returns "certificates.k8s.io"

## Design Notes

- Returns error if Authorizer is nil (required for PodCertificateRequest)
- ClusterTrustBundle available in both v1alpha1 and v1beta1
- PodCertificateRequest only available in v1beta1
- Feature gated resources log warnings when feature gates are disabled
- Note: When adding versions, also update aggregator.go priorities
