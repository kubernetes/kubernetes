# Package: storagemigration

## Purpose
Implements the registry strategy for StorageVersionMigration objects. StorageVersionMigration is a cluster-scoped resource that triggers migration of stored data to a new API version.

## Key Types

- **strategy**: Implements REST strategy for StorageVersionMigration spec operations.
- **statusStrategy**: Extends strategy for status subresource operations.

## Key Variables

- **Strategy**: Singleton for spec operations.
- **StatusStrategy**: Singleton for status subresource operations.

## Key Functions

- **NamespaceScoped()**: Returns false - StorageVersionMigration is cluster-scoped.
- **GetResetFields()**: For spec updates, status is reset; for status updates, metadata/spec are reset.
- **PrepareForCreate(ctx, obj)**: Clears status.
- **Validate(ctx, obj)**: Validates using svmvalidation.ValidateStorageVersionMigration.
- **PrepareForUpdate(ctx, obj, old)**: Preserves status field.
- **ValidateUpdate(ctx, obj, old)**: Validates updates.
- **AllowUnconditionalUpdate()**: Returns false - requires resourceVersion.

## Design Notes

- Part of the StorageVersionMigrator feature for safe API version migrations.
- Status tracks migration progress and completion state.
- Follows standard Kubernetes spec/status separation pattern.
