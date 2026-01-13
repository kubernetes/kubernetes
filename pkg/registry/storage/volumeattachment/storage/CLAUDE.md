# Package: storage

## Purpose
Provides REST storage implementation for VolumeAttachment objects and their status subresource.

## Key Types

- **VolumeAttachmentStorage**: Container holding REST and StatusREST storage.
- **REST**: Wraps genericregistry.Store for VolumeAttachment main resource.
- **StatusREST**: Implements the /status subresource endpoint.

## Key Functions

- **NewStorage(optsGetter)**: Creates REST storage for VolumeAttachment:
  - Uses volumeattachment.Strategy for main operations
  - Uses volumeattachment.StatusStrategy for status operations
  - Returns both VolumeAttachment and Status REST objects
  - Uses ResetFieldsStrategy for proper field isolation
  - Includes TableConvertor for kubectl output formatting

- **StatusREST.Get/Update**: Standard status subresource operations.
- **StatusREST.GetResetFields()**: Returns fields reset on status updates.
