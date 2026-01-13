# Package: validatingwebhookconfiguration

This package provides the registry strategy for ValidatingWebhookConfiguration resources.

## Key Types

- `validatingWebhookConfigurationStrategy` - Validation and defaulting logic

## Key Functions

- `Strategy` - Package-level singleton strategy
- `PrepareForCreate()` - Sets generation to 1
- `PrepareForUpdate()` - Increments generation when webhooks change
- `Validate()` - Validates webhook configuration
- `ValidateUpdate()` - Validates configuration updates

## Design Notes

- Cluster-scoped resource
- Generation increments only when Webhooks array changes
- Does not allow creation via PUT
- Validates webhook URLs, CA bundles, and match rules
- Used by admission controller to call external validation webhooks
- Webhooks can reject or allow requests but cannot modify them
- Supports failure policies (Fail/Ignore) and side effects declaration
