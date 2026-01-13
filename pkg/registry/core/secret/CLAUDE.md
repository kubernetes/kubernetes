# Package: secret

## Purpose
Provides the registry interface and REST strategy implementation for storing Secret API objects with type-specific validation.

## Key Types

- **strategy**: Main strategy for Secret CRUD operations (namespace-scoped).

## Key Functions

- **Strategy** (var): Default logic for creating/updating Secrets.
- **NamespaceScoped()**: Returns `true` - Secrets are namespace-scoped.
- **PrepareForCreate()**: No-op for Secrets.
- **PrepareForUpdate()**: No-op for Secrets.
- **Validate()**: Validates Secret with type-specific rules.
- **WarningsOnCreate()**: Returns TLS validation warnings for kubernetes.io/tls secrets.
- **WarningsOnUpdate()**: Returns TLS validation warnings for kubernetes.io/tls secrets.
- **GetAttrs()**: Returns labels and selectable fields including type.
- **Matcher()**: Returns selection predicate with type field indexing.
- **SelectableFields()**: Returns filterable fields including `type`.

## Design Notes

- Namespace-scoped resource with no subresources.
- Field indexing on `type` for efficient queries by secret type.
- TLS secrets get additional validation warnings (cert/key validation).
- Uses tls.X509KeyPair for TLS certificate validation in warnings.
