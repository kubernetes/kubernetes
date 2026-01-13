# Package: rbac

## Purpose
Defines the internal (unversioned) API types for the rbac.authorization.k8s.io API group, implementing Role-Based Access Control for Kubernetes.

## Key Types

### PolicyRule
Describes what actions are allowed on which resources:
- `Verbs`: Actions allowed (get, list, create, update, delete, etc.). Use "*" for all.
- `APIGroups`: API groups the rule applies to. "" = core, "*" = all.
- `Resources`: Resources the rule applies to. "*" = all, "*/foo" = subresource.
- `ResourceNames`: Optional whitelist of specific resource names.
- `NonResourceURLs`: Non-resource URL paths (cluster-scoped only).

### Subject
Identifies who a binding applies to:
- `Kind`: "User", "Group", or "ServiceAccount"
- `APIGroup`: "" for ServiceAccount, "rbac.authorization.k8s.io" for User/Group
- `Name`: Subject name
- `Namespace`: Required for ServiceAccount

### Role / ClusterRole
- `Role`: Namespaced set of PolicyRules
- `ClusterRole`: Cluster-scoped set of PolicyRules, with optional `AggregationRule`
- `AggregationRule`: Dynamically aggregates rules from other ClusterRoles via label selectors

### RoleBinding / ClusterRoleBinding
- `RoleBinding`: Grants Role or ClusterRole permissions within a namespace
- `ClusterRoleBinding`: Grants ClusterRole permissions cluster-wide
- Both contain `Subjects` and immutable `RoleRef`

## Key Constants
- `APIGroupAll`, `ResourceAll`, `VerbAll`, `NonResourceAll`: Wildcard values ("*")
- `GroupKind`, `ServiceAccountKind`, `UserKind`: Subject kind values
- `AutoUpdateAnnotationKey`: Controls auto-reconciliation of bootstrap roles

## Builder Types
Fluent builders for constructing RBAC objects in code:
- `PolicyRuleBuilder`: Build rules with `NewRule().Groups().Resources().Names().URLs()`
- `ClusterRoleBindingBuilder`: Build bindings with `NewClusterBinding().Groups().Users().SAs()`
- `RoleBindingBuilder`: Build namespaced bindings

## Key Functions
- `ResourceMatches`: Checks if a rule's resources match a request
- `SubjectsStrings`: Extracts users/groups/serviceaccounts from subjects
- `PolicyRule.CompactString()`: Readable string representation for error messages

## Authorization Flow
1. Evaluate ClusterRoleBindings (short-circuit on match)
2. Evaluate RoleBindings in requested namespace (short-circuit on match)
3. Deny by default
