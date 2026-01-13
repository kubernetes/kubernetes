# Package: ensurer

Provides maintenance and reconciliation logic for API Priority and Fairness (APF) configuration objects.

## Key Types

- **EnsureStrategy[ObjectType]**: Generic interface for maintenance strategies (mandatory vs suggested).
- **ObjectOps[ObjectType]**: Combines client operations, cache access, and local object operations.
- **strategy[ObjectType]**: Implements EnsureStrategy with configurable auto-update behavior.

## Key Functions

- **NewMandatoryEnsureStrategy**: Creates a strategy that always overwrites user changes (for system-critical configs).
- **NewSuggestedEnsureStrategy**: Creates a strategy that respects user modifications based on annotations.
- **EnsureConfiguration(s)**: Applies maintenance strategy to create/update APF objects.
- **RemoveUnwantedObjects**: Deletes auto-managed objects that are no longer in the bootstrap set.
- **NewFlowSchemaOps / NewPriorityLevelConfigurationOps**: Factory functions for type-specific operations.

## Design Notes

- Uses Go generics for type-safe operations on FlowSchema and PriorityLevelConfiguration.
- Two maintenance modes:
  - **Mandatory**: Always enforced, user changes are overwritten (e.g., catch-all configurations).
  - **Suggested**: User-editable; uses `apf.kubernetes.io/autoupdate-spec` annotation to track preference.
- Objects with generation=1 and no annotation are assumed safe to auto-update.
- Handles concurrent modifications gracefully with retry logic.
