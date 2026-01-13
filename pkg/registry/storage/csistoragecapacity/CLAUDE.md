# Package: csistoragecapacity

## Purpose
Implements the registry strategy for CSIStorageCapacity objects. CSIStorageCapacity is a namespaced resource that reports storage capacity available for CSI drivers in specific topologies.

## Key Types

- **csiStorageCapacityStrategy**: Implements REST strategy for CSIStorageCapacity CRUD operations.

## Key Variables

- **Strategy**: Singleton instance for Create, Update, and Delete operations.

## Key Functions

- **NamespaceScoped()**: Returns true - CSIStorageCapacity is namespaced.
- **PrepareForCreate(ctx, obj)**: No-op.
- **Validate(ctx, obj)**: Validates using validation.ValidateCSIStorageCapacity.
- **PrepareForUpdate(ctx, obj, old)**: No-op.
- **ValidateUpdate(ctx, obj, old)**: Validates updates, allows invalid label values if already present.
- **AllowUnconditionalUpdate()**: Returns false - requires resourceVersion.
- **WarningsOnCreate/Update**: Returns warnings from storage API utilities.

## Design Notes

- Used by the scheduler to make storage-aware scheduling decisions.
- Reports available capacity per storage class per topology segment.
