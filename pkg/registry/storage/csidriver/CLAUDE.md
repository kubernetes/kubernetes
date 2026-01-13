# Package: csidriver

## Purpose
Implements the registry strategy for CSIDriver objects. CSIDriver is a cluster-scoped resource that contains information about a Container Storage Interface (CSI) driver installed in the cluster.

## Key Types

- **csiDriverStrategy**: Implements REST strategy for CSIDriver CRUD operations.

## Key Variables

- **Strategy**: Singleton instance for Create, Update, and Delete operations.

## Key Functions

- **NamespaceScoped()**: Returns false - CSIDriver is cluster-scoped.
- **PrepareForCreate(ctx, obj)**: No-op, CSIDriver has no mutable status.
- **Validate(ctx, obj)**: Validates using validation.ValidateCSIDriver.
- **PrepareForUpdate(ctx, obj, old)**: No-op.
- **ValidateUpdate(ctx, obj, old)**: Validates updates.
- **AllowUnconditionalUpdate()**: Returns false - requires resourceVersion.
- **WarningsOnCreate/Update**: Returns warnings from storage API utilities.

## Design Notes

- CSIDriver objects describe driver capabilities like volume lifecycle modes, supported access modes, and fsGroup policy.
- Typically created by CSI driver deployments, not by end users.
