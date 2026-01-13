# Package: v1alpha1

Provides versioned defaults for ResourceClaim controller configuration.

## Key Functions

- **RecommendedDefaultResourceClaimControllerConfiguration**: Sets recommended defaults:
  - `ConcurrentResourceClaimSyncs`: 10

## Design Patterns

- Explicit defaulting function pattern.
- Higher default (10) compared to other controllers due to DRA workload patterns.
