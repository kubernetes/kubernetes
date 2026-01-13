# Package: serviceaccount

Implements controllers for managing ServiceAccounts and their token secrets in Kubernetes.

## Key Types

- **ServiceAccountsController**: Ensures default ServiceAccount exists in every namespace.
- **TokensController**: Manages ServiceAccount token secrets (legacy token creation).
- **LegacySATokenCleaner**: Cleans up unused auto-generated legacy SA tokens.

## Key Functions (ServiceAccountsController)

- **NewServiceAccountsController**: Creates controller ensuring service accounts exist.
- **Run**: Starts workers processing namespace events.
- **syncNamespace**: Creates missing ServiceAccounts in active namespaces.

## Key Functions (TokensController)

- **NewTokensController**: Creates controller managing token secrets.
- **Run**: Starts workers for ServiceAccount and Secret processing.
- **syncServiceAccount**: Generates tokens for service accounts when needed.
- **syncSecret**: Validates and populates ServiceAccountToken secrets.

## Key Functions (LegacySATokenCleaner)

- **NewLegacySATokenCleaner**: Creates cleaner for stale auto-generated tokens.
- **Run**: Periodically checks and deletes unused legacy tokens.

## Design Patterns

- ServiceAccountsController uses namespace events to trigger SA creation.
- TokensController uses separate queues for SAs and secrets.
- Tokens are generated using a TokenGenerator interface (RSA-based JWT).
- Supports both auto-generated tokens (legacy) and bound service account tokens.
- Cleans up tokens not used within configured cleanup period.
