# Package: role

Implements the API server registry strategy for Role resources.

## Key Types

- **strategy**: Implements create/update/delete strategies for Role objects.
- **Registry**: Interface for Role storage operations (GetRole).
- **AuthorizerAdapter**: Adapts Registry to the authorizer's RoleGetter interface.

## Key Functions

- **PrepareForCreate / PrepareForUpdate**: Minimal preparation.
- **Validate / ValidateUpdate**: Validates Role objects.
- **NewRegistry**: Creates a Registry from StandardStorage.
- **GetRole**: Retrieves and converts Role to v1 API version.

## Design Notes

- Roles are namespace-scoped resources (unlike ClusterRoles).
- Allows create-on-update and unconditional updates.
- Simpler than ClusterRole (no aggregation rules to handle).
- AuthorizerAdapter adds namespace parameter to GetRole calls.
