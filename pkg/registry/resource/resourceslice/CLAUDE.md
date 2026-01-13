# Package: resourceslice

## Purpose
Implements the registry strategy for ResourceSlice objects in the DRA API. ResourceSlice is a cluster-scoped resource that represents a set of devices published by a DRA driver.

## Key Types

- **resourceSliceStrategy**: Implements REST strategy for ResourceSlice CRUD operations.

## Key Variables

- **Strategy**: Singleton instance for Create, Update, and Delete operations.

## Key Functions

- **NamespaceScoped()**: Returns false - ResourceSlice is cluster-scoped.
- **PrepareForCreate(ctx, obj)**: Sets Generation to 1 and drops disabled feature fields.
- **Validate(ctx, obj)**: Validates using validation.ValidateResourceSlice.
- **PrepareForUpdate(ctx, obj, old)**: Drops disabled fields and increments Generation on spec changes.
- **ValidateUpdate(ctx, obj, old)**: Validates updates.
- **Match(label, field)**: Returns a SelectionPredicate for filtering.
- **GetAttrs(obj)**: Returns labels and selectable fields (metadata.name, nodeName, driverName).

## Feature Gating

- **DRADeviceTaints**: Controls perDeviceNodeSelection and taints fields.
- **DRAConsumableCapacity**: Controls consumedCapacity and sharedCounterSets fields.
