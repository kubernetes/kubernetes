# Package: validatingadmissionpolicy

This package provides the registry strategy for ValidatingAdmissionPolicy resources, which define CEL-based validation rules.

## Key Types

- `validatingAdmissionPolicyStrategy` - Main strategy for spec updates
- `validatingAdmissionPolicyStatusStrategy` - Strategy for status subresource updates

## Key Functions

- `NewStrategy()` - Creates strategy with authorizer and resource resolver
- `NewStatusStrategy()` - Creates status-only update strategy
- `PrepareForCreate()` - Clears status, sets generation to 1
- `PrepareForUpdate()` - Preserves status, increments generation on spec changes
- `Validate()` - Validates policy and authorizes paramKind access
- `GetResetFields()` - Returns fields reset on update (status for spec, spec for status)

## Design Notes

- Cluster-scoped resource
- Separates spec and status update paths
- Status updates don't modify spec or metadata
- Spec updates don't modify status
- Generation only increments on spec changes
- Validates paramKind authorization on create/update
