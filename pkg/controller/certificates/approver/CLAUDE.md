# Package: approver

## Purpose
Implements an automated CSR approver for kubelet client certificates using SubjectAccessReview authorization.

## Key Types/Structs
- `sarApprover`: Approver that uses SubjectAccessReview to authorize CSR approval
- `csrRecognizer`: Struct containing recognize function, required permission, and success message

## Key Functions
- `NewCSRApprovingController(ctx, client, csrInformer)`: Creates a new CSR approving controller
- `recognizers()`: Returns list of CSR recognizers for different certificate types

## Recognized CSR Types
- `selfnodeclient`: Node requesting its own client certificate
- `nodeclient`: Node client certificate requests

## Approval Logic
1. Skip CSRs that already have a certificate or are approved/denied
2. Parse the x509 certificate request
3. Match against recognizers (selfnodeclient, nodeclient patterns)
4. Perform SubjectAccessReview to check if requester can create CSRs with that subresource
5. If authorized, approve the CSR with the recognizer's success message

## Design Notes
- Uses SubjectAccessReview for fine-grained RBAC-based approval decisions
- Builds on the abstract CertificateController base
- Only auto-approves specific well-known CSR patterns from kubelets
- Other signerNames require manual approval or custom approvers
