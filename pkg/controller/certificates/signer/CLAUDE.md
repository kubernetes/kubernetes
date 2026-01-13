# Package: signer

## Purpose
Implements CSR signing controllers that sign approved certificate requests using CA keys stored on local disk.

## Key Types/Structs
- `CSRSigningController`: Wraps CertificateController with dynamic cert reloader
- `signer`: Internal signer with CA provider, client, TTL, and verification function

## Key Functions
- `NewKubeletServingCSRSigningController`: Signs kubernetes.io/kubelet-serving CSRs
- `NewKubeletClientCSRSigningController`: Signs kubernetes.io/kube-apiserver-client-kubelet CSRs
- `NewKubeAPIServerClientCSRSigningController`: Signs kubernetes.io/kube-apiserver-client CSRs
- `NewLegacyUnknownCSRSigningController`: Signs kubernetes.io/legacy-unknown CSRs
- `NewCSRSigningController(ctx, name, signerName, client, informer, caFile, caKeyFile, ttl)`: Generic constructor

## Verification Functions
- `isKubeletServing`: Validates kubelet serving certificate requests
- `isKubeletClient`: Validates kubelet client certificate requests
- `isKubeAPIServerClient`: Validates API server client requests (requires client auth usage)
- `isLegacyUnknown`: No restrictions for legacy compatibility

## Signing Logic
1. Skip unapproved or failed CSRs
2. Match signerName to controller's signer
3. Validate CSR against signer-specific rules
4. Sign using CA with PermissiveSigningPolicy
5. Update CSR status with signed certificate

## Design Notes
- Each signer name has a dedicated controller instance
- CA files are watched for changes via dynamicCertReloader
- TTL can be shortened by CSR's spec.expirationSeconds (minimum 10 min)
- 5-minute backdating for clock skew tolerance
