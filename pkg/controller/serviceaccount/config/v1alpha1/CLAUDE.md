# Package: v1alpha1

Provides versioned defaults for ServiceAccount controller configuration.

## Key Functions

- **RecommendedDefaultSAControllerConfiguration**: Sets recommended defaults:
  - `ConcurrentSATokenSyncs`: 5

- **RecommendedDefaultLegacySATokenCleanerConfiguration**: Sets recommended defaults:
  - `CleanUpPeriod`: 8760 hours (365 days / 1 year)

## Design Patterns

- Explicit defaulting function pattern.
- One-year cleanup period is conservative to avoid deleting tokens still in use.
