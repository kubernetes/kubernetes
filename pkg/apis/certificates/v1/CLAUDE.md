# Package: certificates/v1

## Purpose
Provides defaulting functions and scheme registration for the certificates/v1 API version, which is the stable version for CertificateSigningRequests.

## Key Functions
- **addDefaultingFuncs()**: Registers defaulting functions with the scheme (calls RegisterDefaults)

## Design Notes
- v1 has minimal defaulting as most fields are required or have no sensible defaults
- Unlike v1beta1, v1 requires explicit signerName (no defaulting from CSR content)
- CSR usages and signerName must be explicitly specified by the client
- This is the preferred stable version for certificate signing requests
