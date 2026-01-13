# Package: resourcequota

Implements the ResourceQuota controller that tracks and enforces resource quotas in namespaces.

## Key Types

- **Controller**: Main controller for syncing ResourceQuota objects.
- **QuotaMonitor**: Watches resources and triggers quota recalculation when changes occur.
- **quotaEvaluator**: Evaluates and updates quota usage status.

## Key Functions

- **NewController**: Creates the quota controller with required informers and evaluators.
- **Run**: Starts the controller and quota monitor workers.
- **syncResourceQuota**: Syncs a single ResourceQuota, calculating current usage.
- **replenishQuota**: Triggered by resource changes to update affected quotas.
- **SyncMonitors**: Starts/stops monitors based on discovered API resources.

## Key Features

- Tracks usage of various resource types (pods, services, PVCs, etc.).
- Supports scoped quotas (e.g., BestEffort, NotBestEffort, Terminating pods).
- Uses discovery to find quotable resources dynamically.
- Handles resource additions, deletions, and updates.

## Design Patterns

- Separates controller (processes quotas) from monitor (watches resources).
- Uses quota evaluators for resource-specific usage calculation.
- Implements missingUsageReplenishment for handling resources not yet counted.
- Supports ignoring specific resources based on configuration.
- Emits events for quota calculation errors.
