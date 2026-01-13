# Package: validation

## Purpose
Provides validation logic for all RBAC types: Role, ClusterRole, RoleBinding, and ClusterRoleBinding.

## Key Types

### ClusterRoleValidationOptions
- `AllowInvalidLabelValueInSelector`: Backward compatibility for invalid label values in aggregation selectors

## Key Functions

### Name Validation
- `ValidateRBACName`: Minimal name validation using path segment rules (exported for reuse)

### Role Validation
- `ValidateRole`: Validates namespaced Role objects and their PolicyRules
- `ValidateRoleUpdate`: Validates Role updates including metadata changes
- `ValidateClusterRole`: Validates cluster-scoped ClusterRole with optional aggregation rules
- `ValidateClusterRoleUpdate`: Validates ClusterRole updates

### PolicyRule Validation
- `ValidatePolicyRule(rule, isNamespaced, fldPath)`: Validates a single PolicyRule (exported for reuse)
  - Requires at least one verb
  - NonResourceURLs: Not allowed in namespaced roles, can't mix with resource rules
  - Resource rules: Require apiGroups and resources

### Binding Validation
- `ValidateRoleBinding`: Validates RoleBinding (can reference Role or ClusterRole)
- `ValidateRoleBindingUpdate`: Validates updates, ensures RoleRef is immutable
- `ValidateClusterRoleBinding`: Validates ClusterRoleBinding (must reference ClusterRole)
- `ValidateClusterRoleBindingUpdate`: Validates updates, ensures RoleRef is immutable

### Subject Validation
- `ValidateRoleBindingSubject(subject, isNamespaced, fldPath)`: Validates a Subject (exported for reuse)
  - ServiceAccount: Validates name, requires empty APIGroup, namespace required for cluster bindings
  - User/Group: Requires APIGroup = "rbac.authorization.k8s.io"
  - Name is always required

## Validation Rules
- RoleRef.APIGroup must be "rbac.authorization.k8s.io"
- RoleBinding can reference Role or ClusterRole
- ClusterRoleBinding can only reference ClusterRole
- RoleRef is immutable after creation
- AggregationRule requires at least one selector if present
