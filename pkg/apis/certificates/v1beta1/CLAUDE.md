# Package: certificates/v1beta1

## Purpose
Provides defaulting functions for the deprecated certificates/v1beta1 API version with automatic signer name inference from CSR content.

## Key Defaulting Functions

### SetDefaults_CertificateSigningRequestSpec
- **Usages**: Defaults to [DigitalSignature, KeyEncipherment] if not specified
- **SignerName**: Auto-detected from CSR content using DefaultSignerNameFromSpec()

### SetDefaults_CertificateSigningRequestCondition
- **Status**: Defaults to ConditionTrue if empty

## Key Functions
- **DefaultSignerNameFromSpec()**: Infers signer name by parsing the CSR:
  - Returns KubeAPIServerClientKubeletSignerName if CSR matches kubelet client pattern
  - Returns KubeletServingSignerName if CSR matches kubelet serving pattern
  - Returns LegacyUnknownSignerName otherwise or on parse error

- **IsKubeletServingCSR()**: Checks if CSR matches kubelet serving certificate pattern
- **IsKubeletClientCSR()**: Checks if CSR matches kubelet client certificate pattern
- **usagesToSet()**: Converts usage slice to string set for comparison

## Design Notes
- v1beta1 provides backward compatibility with automatic signer detection
- This behavior differs from v1 which requires explicit signerName
- Deprecated; use certificates/v1 with explicit signer names
- Delegates to internal certificates package for CSR pattern matching
