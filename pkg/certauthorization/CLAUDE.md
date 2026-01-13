# Package: certauthorization

## Purpose
Provides authorization helpers for certificate signer name access control in the certificates.k8s.io API group.

## Key Functions
- `IsAuthorizedForSignerName(ctx, authz, info, verb, signerName)`: Checks if a user is authorized to perform a verb on a signer resource

## Authorization Logic
1. First checks explicit permission for the exact signerName (e.g., "kubernetes.io/kube-apiserver-client")
2. If not allowed, checks wildcard permission for the domain portion (e.g., "kubernetes.io/*")

## Helper Functions
- `buildAttributes`: Creates authorizer.AttributesRecord for a specific signerName
- `buildWildcardAttributes`: Creates attributes for domain wildcard check (e.g., "domain/*")

## Design Notes
- Uses synthetic "signers" resource in certificates.k8s.io API group
- Enables granting permission to all signers under a domain via wildcard RBAC rules
- Supports verbs like "approve", "sign" for certificate signing requests
- APIVersion is set to "*" to match any version
