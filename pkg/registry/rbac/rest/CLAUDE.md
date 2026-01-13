# Package: rest

Provides the REST storage provider for the `rbac.authorization.k8s.io` API group with bootstrap policy initialization.

## Key Types

- **RESTStorageProvider**: Implements storage provider and PostStartHookProvider for RBAC resources.
- **PolicyData**: Holds bootstrap ClusterRoles, ClusterRoleBindings, Roles, and RoleBindings.

## Key Functions

- **NewRESTStorage**: Creates APIGroupInfo with all RBAC resource storage (wrapped with policybased).
- **PostStartHook**: Returns hook that bootstraps RBAC policies after API server starts.
- **EnsureRBACPolicy**: Reconciles bootstrap roles and bindings using component-helpers reconciliation.
- **primeAggregatedClusterRoles**: Migrates legacy roles to aggregated roles.
- **primeSplitClusterRoleBindings**: Handles ClusterRoleBinding splitting migrations.

## Design Notes

- All RBAC storage is wrapped with policybased storage for escalation prevention.
- Creates `DefaultRuleResolver` wiring up all four RBAC registries for authorization checks.
- PostStartHook `rbac/bootstrap-roles` runs at startup with 30-second timeout.
- Bootstrap reconciles ClusterRoles, ClusterRoleBindings, namespaced Roles, and RoleBindings.
- Handles role migrations (aggregation, binding splits) for Kubernetes version upgrades.
- Uses retry logic with backoff for conflict and service-unavailable errors.
