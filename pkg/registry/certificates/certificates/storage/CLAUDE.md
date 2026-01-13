# Package: storage

## Purpose
Provides REST storage implementation for CertificateSigningRequest (CSR) resources with status and approval subresources.

## Key Types

- **REST**: Main REST storage for CSR CRUD operations
- **StatusREST**: REST endpoint for /status subresource (certificate issuance)
- **ApprovalREST**: REST endpoint for /approval subresource (approve/deny)

## Key Functions

- **NewREST(optsGetter)**: Creates REST, StatusREST, and ApprovalREST instances
- **ShortNames()**: Returns ["csr"] for kubectl
- **StatusREST.Update()**: Updates certificate field (uses StatusStrategy)
- **ApprovalREST.Update()**: Updates conditions (uses ApprovalStrategy)
- **countCSRDurationMetric()**: BeginUpdate hook on statusStore for metrics

## Design Notes

- Implements ShortNamesProvider ("csr")
- Three REST endpoints: main resource, /status, and /approval
- /status is for certificate issuance by signers
- /approval is for approve/deny decisions by approval controllers
- Status updates trigger duration metrics (csrDurationRequested, csrDurationHonored)
- Both StatusREST and ApprovalREST support Patch operations
- Shares underlying store between all three endpoints
