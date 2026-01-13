# Package: metrics

Prometheus metrics for the garbage collector controller.

## Key Metrics

- `garbagecollector_controller_resources_sync_error_total`: Counter tracking the number of resource sync errors

## Key Functions

- `Register()`: Registers metrics with the legacy registry (called once via sync.Once)

## Purpose

Provides observability into the garbage collector's operation, specifically tracking errors when syncing resources.

## Design Notes

- Uses the `garbagecollector_controller` subsystem
- Currently only tracks sync errors; additional metrics could be added
- Stability level is ALPHA
