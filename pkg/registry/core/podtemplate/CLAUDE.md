# Package: podtemplate

## Purpose
Provides the registry interface and REST strategy implementation for storing PodTemplate API objects.

## Key Types

- **podTemplateStrategy**: Main strategy for PodTemplate CRUD operations (namespace-scoped).

## Key Functions

- **Strategy** (var): Default logic for creating/updating PodTemplates.
- **NamespaceScoped()**: Returns `true` - PodTemplates are namespace-scoped.
- **PrepareForCreate()**: Sets generation=1.
- **PrepareForUpdate()**: Increments generation if template changes.
- **Validate()**: Validates PodTemplate spec.
- **GetAttrs()**: Returns labels and selectable fields.
- **MatchPodTemplate()**: Returns selection predicate for filtering.

## Design Notes

- Simple strategy with generation tracking.
- Template changes trigger generation increment.
- No subresources.
