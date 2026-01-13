# Package: rolebinding

Implements the API server registry strategy for RoleBinding resources.

## Key Types

- **strategy**: Implements create/update/delete strategies for RoleBinding objects.
- **Registry**: Interface for RoleBinding storage (ListRoleBindings).
- **AuthorizerAdapter**: Adapts Registry to the authorizer's RoleBindingLister interface.

## Key Functions

- **PrepareForCreate / PrepareForUpdate**: Minimal preparation.
- **Validate / ValidateUpdate**: Validates using declarative validation with migration checks.
- **NewRegistry**: Creates a Registry from StandardStorage.
- **ListRoleBindings**: Lists and converts to v1 API version.

## Design Notes

- RoleBindings are namespace-scoped resources.
- Allows create-on-update and unconditional updates.
- Binds Roles or ClusterRoles to users/groups/serviceaccounts within a namespace.
- Uses declarative validation with migration checks.
