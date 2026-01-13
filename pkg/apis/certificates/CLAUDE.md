# Package: certificates

## Purpose
Defines internal (unversioned) types for the Kubernetes certificates API group, managing certificate signing requests and trust bundles.

## Key Types
- **CertificateSigningRequest (CSR)**: Request for a signed certificate from the cluster
- **CertificateSigningRequestSpec**: Contains the PEM-encoded CSR, signer name, usages, expiration
- **CertificateSigningRequestStatus**: Approval conditions and issued certificate
- **ClusterTrustBundle**: Cluster-scoped trust anchor bundle for X.509 certificates
- **PodCertificateRequest**: Per-pod certificate request bound to pod lifecycle

## Key Constants
- **Signer Names**: KubeAPIServerClientSignerName, KubeAPIServerClientKubeletSignerName, KubeletServingSignerName, LegacyUnknownSignerName
- **Key Usages**: Signing, Encryption, ServerAuth, ClientAuth, CodeSigning, etc.
- **Request Conditions**: Approved, Denied, Failed

## Key Functions
- **ParseCSR()**: Parses PEM-encoded CSR from request bytes
- **IsKubeletClientCSR()**: Checks if CSR matches kubelet client certificate pattern
- **IsKubeletServingCSR()**: Checks if CSR matches kubelet serving certificate pattern

## Design Notes
- CSR workflow: create request -> approve/deny -> signer issues certificate
- ClusterTrustBundle provides cluster-wide CA trust distribution
- PodCertificateRequest binds certificate lifecycle to pod lifecycle
- Supports multiple built-in signers with well-known signer names
