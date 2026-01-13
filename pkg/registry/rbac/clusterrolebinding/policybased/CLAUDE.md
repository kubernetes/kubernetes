# Package: policybased

Implements privilege escalation prevention for ClusterRoleBinding storage operations.

## Key Types

- **Storage**: Wraps StandardStorage with escalation checks for bindings.

## Key Functions

- **NewStorage**: Creates Storage with authorizer and rule resolver.
- **Create**: Verifies user can bind the referenced role before creating.
- **Update**: Checks binding authorization on updates; allows GC-only changes.

## Design Notes

- Security layer preventing unauthorized role bindings.
- Users cannot bind roles granting permissions they don't already have.
- Three bypass mechanisms:
  1. `system:masters` group membership
  2. Explicit "bind" permission on the referenced clusterrole
  3. User already has all permissions in the referenced role's rules
- GC-field-only mutations bypass the check.
- Uses ruleResolver to fetch and evaluate the referenced role's rules.
