# Package: v1alpha1

Provides versioned defaults for ResourceQuota controller configuration.

## Key Functions

- **RecommendedDefaultResourceQuotaControllerConfiguration**: Sets recommended defaults:
  - `ResourceQuotaSyncPeriod`: 5 minutes
  - `ConcurrentResourceQuotaSyncs`: 5

## Design Patterns

- Explicit defaulting function pattern.
- 5-minute sync period balances accuracy with API server load.
