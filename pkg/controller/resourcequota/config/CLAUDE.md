# Package: config

Contains configuration types for the ResourceQuota controller.

## Key Types

- **ResourceQuotaControllerConfiguration**: Configuration struct containing:
  - `ResourceQuotaSyncPeriod`: How often to fully recalculate quota usage.
  - `ConcurrentResourceQuotaSyncs`: Number of quotas synced concurrently.

## Design Patterns

- Part of the componentconfig pattern for kube-controller-manager.
- Sync period controls balance between accuracy and API server load.
