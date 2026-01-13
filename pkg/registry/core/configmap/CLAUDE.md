# Package: configmap

## Purpose
Provides the registry interface and REST strategy implementation for storing ConfigMap API objects in the Kubernetes API server.

## Key Types

- **strategy**: Implements `rest.RESTCreateStrategy` and `rest.RESTUpdateStrategy` for ConfigMap objects. Embeds `runtime.ObjectTyper` and `names.NameGenerator`.

## Key Functions

- **Strategy** (var): Default logic applied when creating/updating ConfigMap objects via the REST API.
- **NamespaceScoped()**: Returns `true` - ConfigMaps are namespace-scoped resources.
- **PrepareForCreate()**: Prepares a ConfigMap for creation, dropping disabled fields.
- **PrepareForUpdate()**: Prepares a ConfigMap for update, dropping disabled fields.
- **Validate()**: Validates a new ConfigMap using core validation.
- **ValidateUpdate()**: Validates ConfigMap updates.
- **GetAttrs()**: Returns labels and fields for filtering purposes.
- **Matcher()**: Returns a selection predicate for label/field selectors.
- **SelectableFields()**: Returns filterable fields from ObjectMeta.

## Design Notes

- Implements the standard Kubernetes registry strategy pattern.
- ConfigMaps allow unconditional updates (`AllowUnconditionalUpdate() = true`).
- Does not allow create-on-update (`AllowCreateOnUpdate() = false`).
