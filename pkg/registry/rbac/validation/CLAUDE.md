# Package: validation

Provides RBAC rule resolution and escalation checking utilities.

## Key Types

- **AuthorizationRuleResolver**: Interface for resolving role rules and computing effective permissions.
- **DefaultRuleResolver**: Standard implementation using RBAC role/binding getters and listers.
- **StaticRoles**: Test implementation using in-memory role/binding lists.
- **RoleGetter / RoleBindingLister / ClusterRoleGetter / ClusterRoleBindingLister**: Interfaces for accessing RBAC objects.

## Key Functions

- **ConfirmNoEscalation**: Core function that verifies user has all permissions in the given rules.
- **NewDefaultRuleResolver**: Creates resolver wiring up role/binding accessors.
- **RulesFor**: Computes all PolicyRules applying to a user in a namespace.
- **VisitRulesFor**: Iterates over rules with detailed source information.
- **GetRoleReferenceRules**: Resolves a RoleRef to its PolicyRules.
- **CompactRules**: Combines simple rules differing only by verb (for readable error messages).

## Design Notes

- Central to RBAC privilege escalation prevention.
- Rule resolution walks ClusterRoleBindings (cluster-wide) then RoleBindings (namespace-scoped).
- Subject matching handles Users, Groups, and ServiceAccounts.
- Uses `validation.Covers()` from component-helpers to check rule coverage.
- Error messages include compact, human-readable descriptions of missing permissions.
- `StaticRoles` enables comprehensive unit testing of authorization logic.
