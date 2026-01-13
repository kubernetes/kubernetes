# Package: volumeattachment

## Purpose
Implements the registry strategy for VolumeAttachment objects. VolumeAttachment is a cluster-scoped resource that tracks attachment of a volume to a node.

## Key Types

- **volumeAttachmentStrategy**: Implements REST strategy for VolumeAttachment spec operations.
- **volumeAttachmentStatusStrategy**: Extends volumeAttachmentStrategy for status subresource operations.

## Key Variables

- **Strategy**: Singleton for spec operations.
- **StatusStrategy**: Singleton for status subresource operations.

## Key Functions

- **NamespaceScoped()**: Returns false - VolumeAttachment is cluster-scoped.
- **GetResetFields()**: For spec updates, status is reset; for status updates, metadata/spec are reset.
- **PrepareForCreate(ctx, obj)**: Clears status (not allowed on create).
- **Validate(ctx, obj)**: Validates using validation.ValidateVolumeAttachment and ValidateVolumeAttachmentV1.
- **PrepareForUpdate(ctx, obj, old)**: Preserves status field.
- **ValidateUpdate(ctx, obj, old)**: Validates updates.
- **AllowUnconditionalUpdate()**: Returns false - requires resourceVersion.

## Feature Gating

- **MutableCSINodeAllocatableCount**: Controls ErrorCode field in attach/detach errors.

## Design Notes

- VolumeAttachment objects are typically managed by the attach/detach controller.
- Status contains attach success/error information updated by CSI drivers.
