# Package: certificates

## Purpose
Provides an abstract base controller for managing Certificate Signing Requests (CSRs), used by approver, signer, and cleaner controllers.

## Key Types/Structs
- `CertificateController`: Base controller with name, client, CSR lister, handler function, and workqueue

## Key Functions
- `NewCertificateController(ctx, name, client, csrInformer, handler)`: Creates a new CSR controller with the given handler
- `Run(ctx, workers)`: Starts the controller with specified worker count
- `IsCertificateRequestApproved(csr)`: Checks if CSR has Approved condition
- `HasTrueCondition(csr, conditionType)`: Checks for any True condition of given type
- `GetCertApprovalCondition(status)`: Returns approved/denied state from conditions

## Controller Pattern
- Uses informer for watching CSR add/update/delete events
- Employs rate-limited workqueue with exponential backoff
- Handler function is called for each CSR requiring processing
- Supports multiple workers for concurrent processing

## Design Notes
- Abstract base that specific controllers (approver, signer, cleaner) build upon
- Handler function signature: `func(ctx, *CertificateSigningRequest) error`
- Rate limiter: 200ms-1000s exponential backoff + 10 qps bucket
- Follows standard Kubernetes controller patterns
