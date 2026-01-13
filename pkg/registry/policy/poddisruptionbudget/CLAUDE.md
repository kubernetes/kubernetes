# Package: poddisruptionbudget

Implements the API server registry strategy for PodDisruptionBudget resources in the `policy` API group.

## Key Types

- **podDisruptionBudgetStrategy**: Implements create/update/delete strategies for PodDisruptionBudget objects.
- **podDisruptionBudgetStatusStrategy**: Separate strategy for status subresource updates.

## Key Functions

- **PrepareForCreate**: Clears status and sets initial generation.
- **PrepareForUpdate**: Preserves status, increments generation on spec changes.
- **Validate / ValidateUpdate**: Validates PDB with options for invalid label selector handling.
- **GetResetFields**: Defines reset fields for policy/v1beta1 and policy/v1.
- **ValidateUpdate (status)**: Version-aware status validation using request context.
- **hasInvalidLabelValueInLabelSelector**: Helper to check for grandfathered invalid selectors.

## Design Notes

- PodDisruptionBudgets are namespace-scoped resources.
- Does NOT allow unconditional updates (requires resource version for consistency).
- Does NOT allow create-on-update (POST required).
- Handles backward compatibility for invalid label values in selectors from older PDBs.
- Supports both policy/v1beta1 and policy/v1 API versions.
