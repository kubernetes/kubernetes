# Package: resourcequota

## Purpose
Provides the registry interface and REST strategy implementation for storing ResourceQuota API objects, including status subresource.

## Key Types

- **resourcequotaStrategy**: Main strategy for ResourceQuota CRUD operations (namespace-scoped).
- **resourcequotaStatusStrategy**: Strategy for /status subresource updates.

## Key Functions

- **Strategy** (var): Default logic for creating/updating ResourceQuotas.
- **StatusStrategy** (var): Strategy for status subresource.
- **NamespaceScoped()**: Returns `true` - ResourceQuotas are namespace-scoped.
- **PrepareForCreate()**: Clears status.
- **PrepareForUpdate()**: Preserves status on spec updates.
- **Validate()**: Validates and returns warnings for unknown resources.
- **GetAttrs()**: Returns labels and selectable fields.
- **MatchResourceQuota()**: Returns selection predicate for filtering.

## Design Notes

- Namespace-scoped resource with status subresource.
- Emits warnings for unknown resource types in hard limits.
- Status strategy preserves spec on status-only updates.
- Uses GetResetFields() to ensure proper field isolation between spec and status.
