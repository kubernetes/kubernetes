# Package: certificates/validation

## Purpose
Provides comprehensive validation for CertificateSigningRequest, ClusterTrustBundle, and PodCertificateRequest resources.

## Key Validation Functions

### CSR Validation
- **ValidateCertificateSigningRequestCreate()**: Full validation for new CSRs
- **ValidateCertificateSigningRequestUpdate()**: Validates CSR updates (status changes, approval)
- **ValidateCertificateSigningRequestStatusUpdate()**: Validates status-only updates
- **ValidateCertificateSigningRequestApprovalUpdate()**: Validates approval condition changes

### ClusterTrustBundle Validation
- **ValidateClusterTrustBundle()**: Validates trust bundle resources
- **ValidateClusterTrustBundleUpdate()**: Validates trust bundle updates

### PodCertificateRequest Validation
- **ValidatePodCertificateRequest()**: Validates pod certificate requests
- **ValidatePodCertificateRequestUpdate()**: Validates updates
- **ValidatePodCertificateRequestStatusUpdate()**: Validates status updates

## Key Validation Rules
- **SignerName**: Required, must be valid domain-prefixed path (max 571 chars)
- **Request**: Must be valid PEM-encoded CSR, parseable by x509
- **Usages**: Must be from allowed set, no duplicates
- **ExpirationSeconds**: Must be positive if specified
- **Username/Groups/Extra**: Immutable after creation
- **Conditions**: Can only add Approved/Denied once, not both
- **Certificate**: Must be valid PEM when set in status
- **ClusterTrustBundle**: Must have valid PEM trust anchors, signer name follows naming rules

## Design Notes
- Uses field.ErrorList for detailed error reporting
- Validates cryptographic content (PEM parsing, X.509 certificate parsing)
- Enforces approval workflow rules (can't approve and deny same CSR)
- Signer name validation supports both built-in and custom signers
