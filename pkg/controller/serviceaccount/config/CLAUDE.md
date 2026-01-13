# Package: config

Contains configuration types for the ServiceAccount controller.

## Key Types

- **SAControllerConfiguration**: Configuration struct containing:
  - `ServiceAccountKeyFile`: PEM-encoded private RSA key file for signing tokens.
  - `ConcurrentSATokenSyncs`: Number of token syncing operations done concurrently.
  - `RootCAFile`: Root CA bundle included in token secrets.

- **LegacySATokenCleanerConfiguration**: Configuration struct containing:
  - `CleanUpPeriod`: Duration since last usage before auto-generated tokens can be deleted.

## Design Patterns

- Part of the componentconfig pattern for kube-controller-manager.
- Key file used for JWT token signing.
- Root CA enables in-cluster clients to verify API server TLS.
