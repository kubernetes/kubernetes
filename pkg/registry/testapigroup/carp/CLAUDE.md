# Package: carp

## Purpose
Implements the registry strategy for Carp objects, which is a test API group resource used for testing the Kubernetes API machinery. Carp is a namespaced resource designed to validate registry patterns.

## Key Types

- **carpStrategy**: Implements REST strategy for Carp spec operations.
- **carpStatusStrategy**: Extends carpStrategy for status subresource operations.

## Key Functions

- **NewStrategy(nsClient)**: Creates strategy with namespace client.
- **NewStatusStrategy(carpStrategy)**: Creates status strategy wrapping the main strategy.
- **NamespaceScoped()**: Returns true - Carp is namespaced.
- **GetResetFields()**: Returns fields to reset per API version.
- **PrepareForCreate(ctx, obj)**: Clears status.
- **Validate(ctx, obj)**: Returns nil (no validation for test resource).
- **PrepareForUpdate(ctx, obj, old)**: Preserves status field.
- **Match(label, field)**: Returns a SelectionPredicate for filtering.
- **GetAttrs(obj)**: Returns labels and selectable fields.

## Design Notes

- Test resource for validating API server registry implementations.
- Follows standard Kubernetes spec/status separation pattern.
- Minimal validation since it's only used for testing.
