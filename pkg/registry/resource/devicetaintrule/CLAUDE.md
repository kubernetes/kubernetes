# Package: devicetaintrule

## Purpose
Implements the registry strategy for DeviceTaintRule objects in the DRA API. DeviceTaintRule is a cluster-scoped resource (v1alpha3) that defines taints to be applied to devices.

## Key Types

- **deviceTaintRuleStrategy**: Implements REST strategy for DeviceTaintRule spec operations.
- **deviceTaintRuleStatusStrategy**: Extends deviceTaintRuleStrategy for status subresource operations.

## Key Variables

- **Strategy**: Singleton for spec operations.
- **StatusStrategy**: Singleton for status subresource operations.

## Key Functions

- **NamespaceScoped()**: Returns false - DeviceTaintRule is cluster-scoped.
- **GetResetFields()**: For spec updates, status is reset; for status updates, spec and metadata are reset.
- **PrepareForCreate(ctx, obj)**: Clears status and sets Generation to 1.
- **PrepareForUpdate(ctx, obj, old)**: Preserves status and increments Generation on spec changes.
- **Validate/ValidateUpdate**: Delegates to validation.ValidateDeviceTaintRule/Update.

## Design Notes

- Uses structured-merge-diff fieldpath for field reset strategies.
- Follows standard Kubernetes spec/status separation pattern.
