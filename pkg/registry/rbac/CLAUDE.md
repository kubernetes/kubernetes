# Package: rbac

Provides shared utility functions for RBAC privilege escalation checking in the `rbac.authorization.k8s.io` API group.

## Key Functions

- **EscalationAllowed**: Checks if the user is a member of `system:masters` (superuser group).
- **RoleEscalationAuthorized**: Checks if user has explicit "escalate" permission on the role resource.
- **BindingAuthorized**: Checks if user has explicit "bind" permission on the referenced role.
- **IsOnlyMutatingGCFields**: Detects if an update only changes GC-related fields (ownerRefs, finalizers).

## Design Notes

- Privilege escalation prevention is a core RBAC security feature.
- Users cannot create/update roles/bindings granting permissions they don't already have.
- Three ways to bypass escalation checks:
  1. Be a member of `system:masters` group
  2. Have explicit "escalate" verb on the role resource
  3. Have explicit "bind" verb on the role being referenced
- GC-only field mutations are allowed without escalation checks (to support garbage collection).
