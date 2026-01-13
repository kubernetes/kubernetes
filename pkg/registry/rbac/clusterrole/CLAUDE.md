# Package: clusterrole

Implements the API server registry strategy for ClusterRole resources.

## Key Types

- **strategy**: Implements create/update/delete strategies for ClusterRole objects.
- **Registry**: Interface for ClusterRole storage operations.
- **AuthorizerAdapter**: Adapts Registry to the authorizer's ClusterRoleGetter interface.

## Key Functions

- **PrepareForCreate / PrepareForUpdate**: Minimal preparation (no special handling).
- **Validate / ValidateUpdate**: Validates using declarative validation with migration checks.
- **hasInvalidLabelValueInLabelSelector**: Checks for grandfathered invalid selectors in aggregation rules.
- **NewRegistry**: Creates a Registry from StandardStorage.
- **GetClusterRole**: Retrieves and converts ClusterRole to v1 API version.

## Design Notes

- ClusterRoles are cluster-scoped (non-namespaced) resources.
- Allows create-on-update (`AllowCreateOnUpdate` returns true).
- Allows unconditional updates (no resource version required).
- Handles backward compatibility for invalid label selector values in aggregation rules.
- Uses declarative validation with migration checks.
