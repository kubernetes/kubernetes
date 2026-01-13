# Package: limitrange

## Purpose
Provides the registry interface and REST strategy implementation for storing LimitRange API objects in the Kubernetes API server.

## Key Types

- **limitrangeStrategy**: Implements REST create/update strategies for LimitRange. Embeds `runtime.ObjectTyper` and `names.NameGenerator`.

## Key Functions

- **Strategy** (var): Default logic for creating/updating LimitRange via REST API.
- **NamespaceScoped()**: Returns `true` - LimitRanges are namespace-scoped.
- **PrepareForCreate()**: Auto-generates a UUID name if none provided.
- **AllowCreateOnUpdate()**: Returns `true` - LimitRanges can be created via update.
- **Validate()**: Validates LimitRange objects.
- **ValidateUpdate()**: Validates LimitRange updates (uses same validation as create).

## Design Notes

- Unique behavior: auto-generates UUID names for LimitRanges without explicit names.
- Allows unconditional updates and create-on-update semantics.
- Both create and update validation use the same `ValidateLimitRange` function.
