# Package: cronjob/metrics

## Purpose
Provides Prometheus metrics for monitoring CronJob controller performance.

## Key Constants
- `CronJobControllerSubsystem`: "cronjob_controller" - metric subsystem name

## Key Metrics
- `CronJobCreationSkew`: Histogram measuring time between scheduled run time and actual Job creation
  - Stability: STABLE
  - Buckets: Exponential (1, 2, 10) - covers 1s to ~1024s range
  - Helps identify scheduling delays

## Key Functions
- `Register()`: Registers metrics with the legacy registry (called once via sync.Once)

## Design Notes
- Uses sync.Once to ensure metrics are registered exactly once
- Registered with component-base legacyregistry
- Creation skew metric is useful for detecting controller lag or scheduling issues
