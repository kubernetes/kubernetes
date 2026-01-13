# Package: validatingadmissionpolicybinding

This package provides the registry strategy for ValidatingAdmissionPolicyBinding resources.

## Key Types

- `validatingAdmissionPolicyBindingStrategy` - Validation and defaulting logic
- `PolicyGetter` - Interface for retrieving referenced ValidatingAdmissionPolicy

## Key Functions

- `NewStrategy()` - Creates strategy with authorizer, policy getter, resolver
- `PrepareForCreate()` - Sets generation to 1
- `PrepareForUpdate()` - Increments generation on spec changes
- `Validate()` - Validates binding and authorizes paramRef access
- `ValidateUpdate()` - Validates updates and re-authorizes paramRef

## Design Notes

- Bindings connect policies to target resources via match conditions
- Cluster-scoped (not namespaced)
- References a ValidatingAdmissionPolicy by name
- Can specify namespace selectors and object selectors
- Validates that user can access paramRef resource
- Policy getter enables cross-resource validation
