# Package: csinode

## Purpose
Implements the registry strategy for CSINode objects. CSINode is a cluster-scoped resource that contains node-specific information about CSI drivers available on each node.

## Key Types

- **csiNodeStrategy**: Implements REST strategy for CSINode CRUD operations.

## Key Variables

- **Strategy**: Singleton instance for Create, Update, and Delete operations.

## Key Functions

- **NamespaceScoped()**: Returns false - CSINode is cluster-scoped.
- **PrepareForCreate(ctx, obj)**: No-op.
- **Validate(ctx, obj)**: Validates using validation.ValidateCSINode.
- **PrepareForUpdate(ctx, obj, old)**: No-op.
- **ValidateUpdate(ctx, obj, old)**: Validates updates.
- **AllowUnconditionalUpdate()**: Returns false - requires resourceVersion.

## Design Notes

- CSINode objects are typically managed by the kubelet, not end users.
- Contains driver topology information and allocatable volume counts per node.
