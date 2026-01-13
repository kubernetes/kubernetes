# Package devicemanager

Package devicemanager implements the Device Plugin framework, managing device plugins that expose hardware resources (GPUs, FPGAs, etc.) to pods.

## Key Types

- `Manager`: Interface for device plugin management and resource allocation
- `DeviceRunContainerOptions`: Container runtime options for devices (envs, mounts, devices, CDI)

## Manager Interface Methods

- `Start`: Starts the device plugin registration service
- `Allocate`: Configures and assigns devices to a container
- `GetDeviceRunContainerOptions`: Returns runtime options for a container's devices
- `GetCapacity`: Returns device capacity, allocatable, and inactive resources
- `GetDevices`: Returns devices assigned to a specific container
- `GetTopologyHints`: Provides NUMA topology hints for scheduling
- `UpdateAllocatedResourcesStatus`: Updates pod status with device health
- `UpdatePluginResources`: Sanitizes NodeInfo allocatable for admission (see below)
- `Updates`: Channel for device status change notifications

## Admission Integration

### UpdatePluginResources (manager.go:407)

Called during pod admission via `pluginResourceUpdateFunc`:

```go
func (m *ManagerImpl) UpdatePluginResources(node *schedulerframework.NodeInfo, attrs *lifecycle.PodAdmitAttributes) error
```

1. Checks if pod has any allocated devices
2. Calls `sanitizeNodeAllocatable(node)` to patch allocatable resources

### sanitizeNodeAllocatable (manager.go:1059)

Ensures NodeInfo.Allocatable reflects already-allocated device resources:

```go
func (m *ManagerImpl) sanitizeNodeAllocatable(node *schedulerframework.NodeInfo)
```

**Purpose**: When devices are allocated but node status hasn't been updated yet, this ensures the scheduler framework sees accurate allocatable values.

**Logic**:
1. For each resource in `m.allocatedDevices`:
2. If allocatable < allocated count, update allocatable to match allocated
3. Clones the Resource object if modifications needed (doesn't mutate original)

**Important for NodeInfo Caching**: This function modifies `node.Allocatable`, so any NodeInfo caching must either:
- Clone NodeInfo before calling `UpdatePluginResources`, or
- Invalidate cache after plugin updates

## Constants

- `endpointStopGracePeriod`: 5 minutes grace period after plugin failure
- `kubeletDeviceManagerCheckpoint`: "kubelet_internal_checkpoint"

## Sub-packages

- `checkpoint`: Checkpoint format for persisting device allocations
- `plugin/v1beta1`: Device plugin gRPC API v1beta1 implementation

## Design Notes

- Device plugins register via Unix socket in /var/lib/kubelet/device-plugins/
- Supports ListAndWatch for dynamic device discovery
- Integrates with topology manager for NUMA-aware allocation
- Checkpoints device allocations for persistence across restarts
- Thread-safe via mutex protection in allocatedDevices access
