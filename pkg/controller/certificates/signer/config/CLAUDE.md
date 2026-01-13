# Package: signer/config

## Purpose
Defines configuration types for the CSR signing controller, specifying CA certificate and key file locations.

## Key Types/Structs
- `CSRSigningControllerConfiguration`: Main configuration struct containing:
  - `ClusterSigningCertFile`: Path to cluster-scoped CA certificate (legacy)
  - `ClusterSigningKeyFile`: Path to cluster-scoped CA private key (legacy)
  - `KubeletServingSignerConfiguration`: Cert/key for kubernetes.io/kubelet-serving
  - `KubeletClientSignerConfiguration`: Cert/key for kubernetes.io/kube-apiserver-client-kubelet
  - `KubeAPIServerClientSignerConfiguration`: Cert/key for kubernetes.io/kube-apiserver-client
  - `LegacyUnknownSignerConfiguration`: Cert/key for kubernetes.io/legacy-unknown
  - `ClusterSigningDuration`: Maximum certificate lifetime

- `CSRSigningConfiguration`: Per-signer configuration with CertFile and KeyFile paths

## Design Notes
- Supports separate CA key pairs for each signer name
- ClusterSigningCertFile/KeyFile are legacy single-signer configuration
- Per-signer configuration takes precedence when specified
- ClusterSigningDuration applies to all signers
- Individual CSRs can request shorter durations via spec.expirationSeconds
