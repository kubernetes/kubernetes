# Package: volumebinding

## Purpose
Implements volume binding for pods with PersistentVolumeClaims. Handles both immediate and late binding (WaitForFirstConsumer) storage classes, and manages PV-PVC binding during scheduling.

## Key Types

### VolumeBinding
The plugin struct implementing:
- PreFilter, Filter, Reserve, PreBind, PreScore, Score, EnqueueExtensions, SignPlugin
- **Binder**: SchedulerVolumeBinder for actual binding operations
- **PVCLister**: For accessing PVCs
- **classLister**: For storage class lookup
- **scorer**: For capacity-based scoring

### stateData
Per-pod scheduling state:
- **allBound**: Whether all volumes are already bound
- **podVolumesByNode**: Volume decisions per node
- **podVolumeClaims**: Pod's volume claim information
- **hasStaticBindings**: Whether pod has static (pre-bound) volumes

### SchedulerVolumeBinder (interface)
Volume binding operations:
- FindPodVolumes, AssumePodVolumes, BindPodVolumes
- RevertAssumedPodVolumes

## Extension Points

### PreFilter
- Checks if pod has any PVCs
- For immediate binding, verifies PVCs are bound
- Skips if all volumes already bound

### Filter
- Checks node topology matches PV requirements
- For late binding, finds available PVs matching PVCs
- Considers node affinity of PVs

### Reserve
- Assumes volume bindings for the node
- Updates binder cache with assumed state

### PreBind
- Actually binds PVs to PVCs via API
- Provisions dynamic volumes if needed
- Waits for binding completion

### Score
- Scores based on storage capacity availability (when enabled)
- Prefers nodes with more available capacity

## Design Pattern
- Two-phase binding: assume during Reserve, commit during PreBind
- Supports both pre-provisioned and dynamically provisioned volumes
- Capacity scoring via StorageCapacityScoring feature gate
- Handles WaitForFirstConsumer delayed binding
