# Package: prioritylevelconfiguration

Implements the API server registry strategy for PriorityLevelConfiguration resources.

## Key Types

- **priorityLevelConfigurationStrategy**: Implements create/update/delete strategies for PriorityLevelConfiguration.
- **priorityLevelConfigurationStatusStrategy**: Separate strategy for status subresource updates.

## Key Functions

- **PrepareForCreate**: Clears status and sets initial generation.
- **PrepareForUpdate**: Increments generation on spec changes, preserves status.
- **Validate / ValidateUpdate**: Validates using version-aware validation options.
- **GetResetFields**: Defines fields reset by server for each API version.
- **getRequestGroupVersion**: Extracts API version from request context for version-specific validation.

## Design Notes

- PriorityLevelConfigurations are cluster-scoped.
- Version-aware validation handles NominalConcurrencyShares field compatibility (1.28+ considerations).
- Supports v1beta1, v1beta2, v1beta3, and v1 API versions.
- Status updates preserve spec and most metadata, only allowing status field changes.
