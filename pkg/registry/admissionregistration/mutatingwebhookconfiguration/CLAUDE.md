# Package: mutatingwebhookconfiguration

This package provides the registry strategy for MutatingWebhookConfiguration resources, which define external webhooks that can modify admission requests.

## Key Types

- `mutatingWebhookConfigurationStrategy` - Implements validation and defaulting logic

## Key Functions

- `Strategy` - Package-level singleton strategy instance
- `PrepareForCreate()` - Sets generation to 1 on creation
- `PrepareForUpdate()` - Increments generation when webhooks change
- `Validate()` - Validates webhook configuration
- `ValidateUpdate()` - Validates webhook configuration updates

## Design Notes

- MutatingWebhookConfiguration is cluster-scoped
- Generation increments when the Webhooks array changes
- Does not allow creation via PUT (AllowCreateOnUpdate = false)
- Validates webhook URLs, client configs, and match rules
- Used by the admission controller to call external mutation webhooks
- Webhooks can modify objects before they are persisted
