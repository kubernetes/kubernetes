# Package: metrics

## Purpose
Defines and registers Prometheus metrics for the Attach/Detach controller to track volume state and forced detach operations.

## Key Constants

- **ForceDetachReasonTimeout**: Force detach due to unmount timeout.
- **ForceDetachReasonOutOfService**: Force detach due to out-of-service taint.

## Key Metrics

- **inUseVolumeMetricDesc**: Gauge measuring number of volumes in use per node and plugin.
- **totalVolumesMetricDesc**: Gauge tracking total volumes in desired/actual state of world per plugin.
- **ForceDetachMetricCounter**: Counter tracking forced detach operations by reason.

## Key Types

- **attachDetachStateCollector**: Custom collector implementing metrics.StableCollector for volume state metrics.
- **volumeCount**: Helper type (map of maps) for counting volumes by node/state and plugin.

## Key Functions

- **Register(...)**: Thread-safe registration of metrics collectors.
- **RecordForcedDetachMetric(reason)**: Increment force detach counter.
- **getVolumeInUseCount**: Counts volumes per node from pod lister.
- **getTotalVolumesCount**: Counts volumes in DSW and ASW.

## Design Notes

- Uses custom collector pattern for dynamic volume state metrics.
- Metrics are registered with the legacy registry for backward compatibility.
- Stability level is ALPHA for all metrics.
