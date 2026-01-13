# Package: policybased

Implements privilege escalation prevention for ClusterRole storage operations.

## Key Types

- **Storage**: Wraps StandardStorage with escalation checks, adding authorizer and rule resolver.

## Key Functions

- **NewStorage**: Creates Storage wrapping the given StandardStorage with escalation prevention.
- **Create**: Checks if user can escalate before creating; verifies user has all permissions in the role's rules.
- **Update**: Checks escalation on updates; allows GC-only field changes without escalation check.
- **hasAggregationRule**: Helper to detect if ClusterRole uses aggregation.

## Design Notes

- Critical security layer: prevents users from creating ClusterRoles with more permissions than they have.
- Aggregation rules require cluster-admin privileges (full `*` on `*.*`) since they can gather any permissions.
- Three bypass mechanisms:
  1. `system:masters` group membership
  2. Explicit "escalate" permission on clusterroles
  3. User already has all permissions in the role's rules
- GC-field-only mutations bypass escalation check (allows garbage collection to proceed).
