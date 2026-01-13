# Package: certificates/fuzzer

## Purpose
Provides fuzz testing functions for the certificates API types to ensure proper serialization roundtrip behavior during API testing.

## Key Functions
- **Funcs()**: Returns fuzzer functions for the certificates API group

## Fuzzer Functions
1. **CertificateSigningRequestSpec fuzzer**: Sets valid defaults:
   - Usages: [KeyEncipherment]
   - SignerName: "example.com/custom-sample-signer"
   - ExpirationSeconds: Converts 1h1m1s duration to seconds

2. **CertificateSigningRequestCondition fuzzer**: Ensures Status is set:
   - Defaults Status to ConditionTrue if empty

3. **PodCertificateRequestSpec fuzzer**: Ensures MaxExpirationSeconds is non-nil:
   - Defaults to 86400 (24 hours) if nil
   - Required because the field has a defaulter

## Design Notes
- Fuzzer ensures fields have valid values that pass validation
- Used by `pkg/api/testing/serialization_test.go` for roundtrip tests
- Prevents nil pointer issues during fuzz testing
- ExpirationSeconds conversion uses client-go's csr.DurationToExpirationSeconds
