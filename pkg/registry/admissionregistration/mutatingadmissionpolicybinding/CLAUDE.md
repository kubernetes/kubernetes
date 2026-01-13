# Package: mutatingadmissionpolicybinding

This package provides the registry strategy for MutatingAdmissionPolicyBinding resources, which bind policies to specific resources and namespaces.

## Key Types

- `mutatingAdmissionPolicyBindingStrategy` - Implements validation and defaulting logic
- `PolicyGetter` - Interface for retrieving the referenced MutatingAdmissionPolicy

## Key Functions

- `NewStrategy()` - Creates a new strategy with authorizer, policy getter, and resource resolver
- `PrepareForCreate()` - Initializes generation to 1
- `PrepareForUpdate()` - Increments generation when spec changes
- `Validate()` - Validates binding spec and authorizes paramRef access
- `ValidateUpdate()` - Validates updates and re-authorizes paramRef if changed

## Design Notes

- Bindings are cluster-scoped (not namespaced)
- References a MutatingAdmissionPolicy by name
- Can specify match conditions to filter which resources the policy applies to
- Validates that the user can access the paramRef resource
- Generation tracks spec changes for controller reconciliation
