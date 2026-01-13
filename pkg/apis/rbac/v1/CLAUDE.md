# Package: v1

## Purpose
Provides the v1 (stable) versioned API registration, defaults, and helper builders for the rbac.authorization.k8s.io API group.

## Key Constants
- `GroupName`: "rbac.authorization.k8s.io"
- `SchemeGroupVersion`: rbac.authorization.k8s.io/v1

## Key Defaulting Functions
- `SetDefaults_ClusterRoleBinding`: Defaults RoleRef.APIGroup to GroupName if empty
- `SetDefaults_RoleBinding`: Defaults RoleRef.APIGroup to GroupName if empty
- `SetDefaults_Subject`: Defaults APIGroup based on Kind:
  - ServiceAccount: ""
  - User/Group: "rbac.authorization.k8s.io"

## Builder Types (for constructing RBAC objects in code)

### PolicyRuleBuilder
Fluent API for building PolicyRules:
- `NewRule(verbs...)`: Start building a rule
- `Groups(groups...)`, `Resources(resources...)`, `Names(names...)`, `URLs(urls...)`
- `Rule()`: Returns rule and error, `RuleOrDie()`: Panics on error
- Validates: verbs required, resource rules need apiGroups, non-resource rules can't mix with resources

### ClusterRoleBindingBuilder / RoleBindingBuilder
- `NewClusterBinding(clusterRoleName)`, `NewRoleBinding(roleName, namespace)`
- `Groups(...)`, `Users(...)`, `SAs(namespace, names...)`
- `Binding()` / `BindingOrDie()`

## Code Generation Markers
- `+k8s:conversion-gen`, `+k8s:defaulter-gen`, `+k8s:deepcopy-gen`, `+k8s:validation-gen`

## Notes
- External types sourced from `k8s.io/api/rbac/v1`
- Builders sort fields for consistent output
- GA (stable) version of RBAC API
