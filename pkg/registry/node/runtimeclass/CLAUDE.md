# Package: runtimeclass

Implements the API server registry strategy for RuntimeClass resources.

## Key Types

- **strategy**: Implements create/update/delete strategies for RuntimeClass objects.

## Key Functions

- **PrepareForCreate / PrepareForUpdate**: Minimal preparation (no special handling).
- **Validate / ValidateUpdate**: Validates using declarative validation with migration checks and node normalization rules.
- **WarningsOnCreate / WarningsOnUpdate**: Returns warnings from `nodeapi.GetWarningsForRuntimeClass`.

## Design Notes

- RuntimeClass is a cluster-scoped (non-namespaced) resource.
- Defines container runtime configuration to be used by pods.
- Allows create-on-update (`AllowCreateOnUpdate` returns true).
- Does NOT allow unconditional updates (requires resource version).
- Uses declarative validation with node-specific normalization rules.
- No status subresource (spec-only resource).
