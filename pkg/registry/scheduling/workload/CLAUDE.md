# Package: workload

## Purpose
Implements the registry strategy for Workload objects in the scheduling API (v1alpha2). Workload is a namespaced resource that represents a schedulable workload unit, enabling gang scheduling and workload-aware scheduling.

## Key Types

- **workloadStrategy**: Implements REST strategy for Workload spec operations.
- **workloadStatusStrategy**: Extends workloadStrategy for status subresource operations.

## Key Variables

- **Strategy**: Singleton for spec operations.
- **StatusStrategy**: Singleton for status subresource operations.

## Key Functions

- **NamespaceScoped()**: Returns true - Workload is namespaced.
- **GetResetFields()**: For spec updates, status is reset; for status updates, spec/metadata are reset.
- **PrepareForCreate(ctx, obj)**: Clears status and sets Generation to 1.
- **PrepareForUpdate(ctx, obj, old)**: Preserves status, increments Generation on spec changes.
- **Match(label, field)**: Returns a SelectionPredicate for filtering.
- **GetAttrs(obj)**: Returns labels and selectable fields.

## Design Notes

- Part of the GenericWorkload feature for gang scheduling.
- Follows standard Kubernetes spec/status separation pattern.
