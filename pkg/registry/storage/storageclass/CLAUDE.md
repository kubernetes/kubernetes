# Package: storageclass

## Purpose
Implements the registry strategy for StorageClass objects. StorageClass is a cluster-scoped resource that defines storage provisioner parameters for dynamic volume provisioning.

## Key Types

- **storageClassStrategy**: Implements REST strategy for StorageClass CRUD operations.

## Key Variables

- **Strategy**: Singleton instance for Create, Update, and Delete operations.

## Key Functions

- **NamespaceScoped()**: Returns false - StorageClass is cluster-scoped.
- **PrepareForCreate(ctx, obj)**: No-op.
- **Validate(ctx, obj)**: Validates using validation.ValidateStorageClass with declarative validation.
- **PrepareForUpdate(ctx, obj, old)**: No-op.
- **ValidateUpdate(ctx, obj, old)**: Validates updates with declarative validation.
- **AllowUnconditionalUpdate()**: Returns true - unconditional updates allowed.
- **WarningsOnCreate/Update**: Returns warnings from storage API utilities.

## Design Notes

- StorageClass defines provisioner, parameters, reclaim policy, and mount options.
- Used by PersistentVolumeClaim to request dynamically provisioned storage.
