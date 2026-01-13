# Package: replicationcontroller

## Purpose
Provides the registry interface and REST strategy implementation for storing ReplicationController API objects, including status subresource.

## Key Types

- **rcStrategy**: Main strategy for RC CRUD operations (namespace-scoped).
- **rcStatusStrategy**: Strategy for /status subresource updates.

## Key Functions

- **Strategy** (var): Default logic for creating/updating RCs.
- **StatusStrategy** (var): Strategy for status subresource.
- **NamespaceScoped()**: Returns `true` - RCs are namespace-scoped.
- **DefaultGarbageCollectionPolicy()**: Returns `rest.OrphanDependents` for backwards compatibility.
- **PrepareForCreate()**: Sets generation=1, clears status.
- **PrepareForUpdate()**: Increments generation on spec changes, preserves status.
- **GetAttrs()**: Returns labels and selectable fields.
- **MatchReplicationController()**: Returns selection predicate for filtering.
- **ToSelectableFields()**: Returns filterable fields including status.replicas.

## Design Notes

- Namespace-scoped resource with status subresource.
- Default GC policy is OrphanDependents for backwards compatibility.
- Generation tracking for spec changes.
- Supports field selection on status.replicas.
