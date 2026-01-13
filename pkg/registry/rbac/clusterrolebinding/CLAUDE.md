# Package: clusterrolebinding

Implements the API server registry strategy for ClusterRoleBinding resources.

## Key Types

- **strategy**: Implements create/update/delete strategies for ClusterRoleBinding objects.
- **Registry**: Interface for ClusterRoleBinding storage (ListClusterRoleBindings).
- **AuthorizerAdapter**: Adapts Registry to the authorizer's ClusterRoleBindingLister interface.

## Key Functions

- **PrepareForCreate / PrepareForUpdate**: Minimal preparation (no special handling).
- **Validate / ValidateUpdate**: Validates ClusterRoleBinding objects.
- **NewRegistry**: Creates a Registry from StandardStorage.
- **ListClusterRoleBindings**: Lists and converts to v1 API version.

## Design Notes

- ClusterRoleBindings are cluster-scoped (non-namespaced) resources.
- Allows create-on-update (`AllowCreateOnUpdate` returns true).
- Allows unconditional updates (no resource version required).
- Binds ClusterRoles to users/groups/serviceaccounts cluster-wide.
