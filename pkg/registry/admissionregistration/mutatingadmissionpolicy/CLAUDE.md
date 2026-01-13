# Package: mutatingadmissionpolicy

This package provides the registry strategy for MutatingAdmissionPolicy resources, which define CEL-based mutation rules for admission control.

## Key Types

- `mutatingAdmissionPolicyStrategy` - Implements validation and defaulting logic for MutatingAdmissionPolicy

## Key Functions

- `NewStrategy()` - Creates a new strategy with authorizer and resource resolver
- `PrepareForCreate()` - Initializes generation to 1 on creation
- `PrepareForUpdate()` - Increments generation when spec changes
- `Validate()` - Validates the policy spec and authorizes paramKind access
- `ValidateUpdate()` - Validates updates and re-authorizes paramKind if changed

## Design Notes

- MutatingAdmissionPolicy is cluster-scoped (not namespaced)
- Validates that the user can access the paramKind resource specified
- Generation is incremented on spec changes (not metadata-only changes)
- Does not allow creation via PUT (AllowCreateOnUpdate = false)
- Works with MutatingAdmissionPolicyBinding to apply mutations
