# Package: policybased

Implements privilege escalation prevention for Role storage operations.

## Key Types

- **Storage**: Wraps StandardStorage with escalation checks for namespace-scoped roles.

## Key Functions

- **NewStorage**: Creates Storage with authorizer and rule resolver.
- **Create**: Verifies user has all permissions in the role's rules before creating.
- **Update**: Checks escalation on updates; allows GC-only changes.

## Design Notes

- Security layer for namespace-scoped Roles.
- Same escalation prevention pattern as ClusterRole but namespace-aware.
- Users cannot create Roles granting permissions they don't have in that namespace.
- Bypass mechanisms: `system:masters` membership or explicit "escalate" permission.
- GC-field-only mutations bypass the check.
