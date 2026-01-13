# Package: flowschema

Implements the API server registry strategy for FlowSchema resources in the `flowcontrol.apiserver.k8s.io` API group.

## Key Types

- **flowSchemaStrategy**: Implements create/update/delete strategies for FlowSchema objects.
- **flowSchemaStatusStrategy**: Separate strategy for status subresource updates.

## Key Functions

- **PrepareForCreate**: Clears status and sets initial generation.
- **PrepareForUpdate**: Increments generation on spec changes, preserves status.
- **Validate / ValidateUpdate**: Validates FlowSchema using flowcontrol validation.
- **GetResetFields**: Defines fields reset by the server (status for main, spec/metadata for status updates).

## Design Notes

- FlowSchemas are cluster-scoped (not namespaced).
- Separate strategies for spec vs status updates prevent cross-contamination.
- Supports multiple API versions (v1beta1, v1beta2, v1beta3, v1) with version-specific reset fields.
- Generation tracking distinguishes spec updates from status-only updates.
