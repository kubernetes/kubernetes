# Package: metrics

## Purpose
Defines and registers Prometheus metrics for the PersistentVolume controller to track PV/PVC counts, operation errors, and operation latencies.

## Key Types

- **PVLister**: Interface for listing PVs (implemented by cache.Store).
- **PVCLister**: Interface for listing PVCs (implemented by cache.Store).
- **pvAndPVCCountCollector**: Custom metrics collector for PV/PVC state.
- **OperationStartTimeCache**: Thread-safe cache for tracking operation start times.

## Key Metrics

- **total_pv_count**: Gauge of total PVs by plugin name and volume mode.
- **bound_pv_count**: Gauge of bound PVs by storage class.
- **unbound_pv_count**: Gauge of unbound PVs by storage class.
- **bound_pvc_count**: Gauge of bound PVCs by namespace, storage class, and volume attributes class.
- **unbound_pvc_count**: Gauge of unbound PVCs by namespace, storage class, and volume attributes class.
- **volume_operation_total_errors**: Counter of volume operation errors by plugin and operation.
- **retroactive_storageclass_total**: Counter of retroactive StorageClass assignments.
- **retroactive_storageclass_errors_total**: Counter of failed retroactive StorageClass assignments.

## Key Functions

- **Register(pvLister, pvcLister, pluginMgr)**: Registers all metrics collectors.
- **RecordVolumeOperationErrorMetric(pluginName, opName)**: Records an operation error.
- **RecordRetroactiveStorageClassMetric(success)**: Records retroactive StorageClass assignment.
- **RecordMetric(key, cache, err)**: Records latency or error based on operation result.

## Design Notes

- Uses custom collector pattern for dynamic PV/PVC count metrics.
- Operation timestamps enable latency tracking for provisioning/deletion.
- All metrics have ALPHA stability level.
