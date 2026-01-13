# Package: volumezone

## Purpose
Implements a filter plugin that ensures pods are scheduled only on nodes in the same zone as their bound PersistentVolumes. Prevents cross-zone volume attachments.

## Key Types

### VolumeZone
The plugin struct implementing:
- PreFilterPlugin, FilterPlugin, EnqueueExtensions, SignPlugin
- **pvLister**: For PV zone label lookup
- **pvcLister**: For PVC to PV mapping
- **scLister**: For storage class zone constraints

### stateData
Per-pod state containing:
- **podPVTopologies**: List of pvTopology for each volume

### pvTopology
Zone information for a PV:
- **pvName**: Name of the PersistentVolume
- **key**: Topology label key (zone or region)
- **values**: Set of allowed zones/regions

## Extension Points

### PreFilter
- Gathers zone requirements from all pod volumes
- Extracts topology labels from bound PVs
- Handles both pre-bound and dynamically provisioned volumes
- Skips if no volumes with zone constraints

### Filter
- Checks if node's zone labels match PV requirements
- Verifies node is in an allowed zone for each PV
- Returns UnschedulableAndUnresolvable if zone mismatch

## Topology Labels Checked
- `topology.kubernetes.io/zone` (GA)
- `topology.kubernetes.io/region` (GA)
- `failure-domain.beta.kubernetes.io/zone` (deprecated)
- `failure-domain.beta.kubernetes.io/region` (deprecated)

## Key Functions

- **New(ctx, obj, handle, features)**: Creates the plugin
- **SignPod(ctx, pod)**: Returns volumes for pod signing
- **translateToGALabel(label)**: Converts beta labels to GA labels
- **EventsToRegister()**: Returns PV/Add, PV/Update, PVC/Add, StorageClass/Add events

## Design Pattern
- Critical for cloud environments with zonal storage
- Prevents scheduling failures due to cross-zone attachment attempts
- Handles label format migration (beta to GA)
- Supports both static and dynamic provisioning
