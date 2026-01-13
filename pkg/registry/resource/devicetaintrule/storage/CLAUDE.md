# Package: storage

## Purpose
Provides REST storage implementation for DeviceTaintRule objects and their status subresource.

## Key Types

- **REST**: Wraps genericregistry.Store for DeviceTaintRule main resource.
- **StatusREST**: Implements the /status subresource endpoint.

## Key Functions

- **NewREST(optsGetter)**: Creates REST storage for DeviceTaintRule, returns both REST and StatusREST:
  - Main store uses devicetaintrule.Strategy
  - Status store uses devicetaintrule.StatusStrategy
  - Both use ResetFieldsStrategy for proper field isolation

- **StatusREST.Get(ctx, name, options)**: Retrieves object for Patch support.
- **StatusREST.Update(...)**: Updates only the status subset, explicitly disallows create-on-update.
- **StatusREST.GetResetFields()**: Returns fields that should be reset on status updates.
