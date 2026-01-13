# Package: deviceclass

## Purpose
Implements the registry strategy for DeviceClass objects in the Dynamic Resource Allocation (DRA) API. DeviceClass is a cluster-scoped resource that defines device types and their configurations.

## Key Types

- **deviceClassStrategy**: Implements REST strategy interfaces for DeviceClass CRUD operations.

## Key Variables

- **Strategy**: Singleton instance of deviceClassStrategy used for Create, Update, and Delete operations.

## Key Functions

- **NamespaceScoped()**: Returns false - DeviceClass is cluster-scoped.
- **PrepareForCreate(ctx, obj)**: Sets Generation to 1 and drops disabled feature fields.
- **Validate(ctx, obj)**: Validates the DeviceClass using validation.ValidateDeviceClass.
- **PrepareForUpdate(ctx, obj, old)**: Drops disabled fields and increments Generation on spec changes.
- **ValidateUpdate(ctx, obj, old)**: Validates updates using validation.ValidateDeviceClassUpdate.
- **AllowUnconditionalUpdate()**: Returns true - unconditional updates are allowed.

## Feature Gating

- **DRAExtendedResource**: When disabled, the ExtendedResourceName field is dropped unless already in use.
