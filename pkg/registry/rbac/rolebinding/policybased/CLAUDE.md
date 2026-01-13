# Package: policybased

Implements privilege escalation prevention for RoleBinding storage operations.

## Key Types

- **Storage**: Wraps StandardStorage with escalation checks for namespace-scoped bindings.

## Key Functions

- **NewStorage**: Creates Storage with authorizer and rule resolver.
- **Create**: Verifies user can bind the referenced role in the target namespace.
- **Update**: Checks binding authorization; allows GC-only changes.

## Design Notes

- Security layer for namespace-scoped RoleBindings.
- Users cannot bind roles granting permissions they don't have in that namespace.
- Namespace is extracted from request context (URL path).
- Can bind either Roles (same namespace) or ClusterRoles (grants cluster-wide role in namespace scope).
- Bypass mechanisms: `system:masters` membership or explicit "bind" permission.
- GC-field-only mutations bypass the check.
