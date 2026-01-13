# Package: event

## Purpose
Provides the registry interface and REST strategy implementation for storing Event API objects in the Kubernetes API server.

## Key Types

- **eventStrategy**: Implements REST create/update strategies for Events. Embeds `runtime.ObjectTyper` and `names.NameGenerator`.

## Key Functions

- **Strategy** (var): Default logic for creating/updating Events via REST API.
- **NamespaceScoped()**: Returns `true` - Events are namespace-scoped.
- **DefaultGarbageCollectionPolicy()**: Returns `rest.Unsupported` - Events don't support garbage collection.
- **AllowCreateOnUpdate()**: Returns `true` - Events can be created via update.
- **Validate()**: Validates Events using group version from request context.
- **ValidateUpdate()**: Validates Event updates with version-aware validation.
- **GetAttrs()**: Returns labels and selectable fields for filtering.
- **Matcher()**: Returns a selection predicate for label/field selectors.
- **ToSelectableFields()**: Returns extensive filterable fields including involvedObject details, reason, source, type, and reportingComponent.
- **requestGroupVersion()**: Extracts API group/version from request context for version-specific validation.

## Design Notes

- Events support rich field selection for filtering by involvedObject (kind, namespace, name, uid, apiVersion, resourceVersion, fieldPath), reason, source, type.
- Source field falls back to reportingController if Component is empty.
- Allows unconditional updates and create-on-update semantics.
