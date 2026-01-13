# Package: namespace

## Purpose
Provides the registry interface and REST strategy implementation for storing Namespace API objects, including status and finalize subresource strategies.

## Key Types

- **namespaceStrategy**: Main strategy for Namespace CRUD operations.
- **namespaceStatusStrategy**: Strategy for /status subresource updates.
- **namespaceFinalizeStrategy**: Strategy for /finalize subresource updates.

## Key Functions

- **Strategy** (var): Default logic for creating/updating Namespaces.
- **StatusStrategy** (var): Strategy for status subresource.
- **FinalizeStrategy** (var): Strategy for finalize subresource.
- **NamespaceScoped()**: Returns `false` - Namespaces are cluster-scoped.
- **GetResetFields()**: Returns fields that should be reset (status for main, spec for status strategy).
- **PrepareForCreate()**: Sets status to Active, ensures kubernetes finalizer is present.
- **PrepareForUpdate()**: Preserves finalizers and status from old object.
- **Canonicalize()**: Ensures `metadata.name` label matches the namespace name (for GenerateName support).
- **GetAttrs()**: Returns labels and selectable fields.
- **MatchNamespace()**: Returns selection predicate for filtering.
- **NamespaceToSelectableFields()**: Returns filterable fields including `status.phase` and `name`.

## Design Notes

- Cluster-scoped resource (not namespace-scoped).
- Implements field reset strategies for server-side apply.
- Auto-adds `kubernetes` finalizer on creation.
- Maintains `kubernetes.io/metadata.name` label synchronized with name.
- Supports filtering by `status.phase` and `name` (legacy compatibility).
