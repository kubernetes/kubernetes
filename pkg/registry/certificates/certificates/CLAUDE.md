# Package: certificates

## Purpose
Implements the registry strategy for CertificateSigningRequest (CSR) resources, which allow clients to request X.509 certificates from the cluster.

## Key Types

- **csrStrategy**: Base strategy for CSR objects
- **csrStatusStrategy**: Strategy for /status subresource (certificate issuance)
- **csrApprovalStrategy**: Strategy for /approval subresource (approve/deny)

## Key Functions

- **Strategy**: Default logic for creating/updating CSRs
- **StatusStrategy**: Logic for status updates (certificate field)
- **ApprovalStrategy**: Logic for approval updates (conditions)
- **PrepareForCreate()**: Injects user info from context (username, uid, groups, extra), clears status
- **PrepareForUpdate()**: CSRs are immutable - preserves spec and status from old object
- **GetAttrs()**: Returns labels and selectable fields (including spec.signerName)
- **SelectableFields()**: Supports filtering by spec.signerName
- **preserveConditionInstances()**: Prevents adding/removing Approved/Denied conditions via status
- **populateConditionTimestamps()**: Auto-sets LastUpdateTime and LastTransitionTime

## Design Notes

- Cluster-scoped resource
- User info is captured from request context on creation (cannot be set by user)
- Status subresource is for issuing certificate, approval subresource is for approve/deny
- Approved/Denied conditions can only be modified via /approval, not /status
- Supports field selector on spec.signerName for filtering
- Uses declarative validation with migration checks
