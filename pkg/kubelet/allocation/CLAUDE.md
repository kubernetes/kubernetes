# Package: allocation

## Purpose
This package handles tracking pod resource allocations for in-place vertical pod scaling. It manages the checkpointing of allocated resources and coordinates resize operations including admission, deferral, and retry logic.

## Key Types

- **Manager**: Interface for tracking and managing pod resource allocations
- **manager**: Implementation with checkpoint state, admission handlers, and resize tracking

## Key Functions

- **NewManager()**: Creates an allocation manager with checkpoint-based persistence
- **NewInMemoryManager()**: Creates an in-memory manager for testing
- **AddPod()**: Admits a pod and updates its resource allocation
- **UpdatePodFromAllocation()**: Overwrites pod spec with stored allocation
- **SetAllocatedResources()**: Checkpoints the resources allocated to a pod
- **handlePodResourcesResize()**: Processes pod resize requests
- **RetryPendingResizes()**: Retries deferred resize operations

## Admission Flow

### canAdmitPod() (line 689)

Determines if a pod can be admitted to the node:

```go
func (m *manager) canAdmitPod(logger klog.Logger, allocatedPods []*v1.Pod, pod *v1.Pod) (bool, string, string)
```

1. Filters out the pod being evaluated from `allocatedPods`
2. Creates `PodAdmitAttributes{Pod: pod, OtherPods: allocatedPods}`
3. Iterates through all registered `admitHandlers`
4. Returns first rejection reason, or success if all pass

### canResizePod() (line 709)

Determines if a requested resize is currently feasible:

```go
func (m *manager) canResizePod(logger klog.Logger, allocatedPods []*v1.Pod, pod *v1.Pod) (bool, string, string)
```

Checks for infeasible resize scenarios:
- Guaranteed QoS pods with CPU Manager static policy (unless `InPlacePodVerticalScalingExclusiveCPUs` enabled)
- Guaranteed QoS pods with Memory Manager static policy (unless `InPlacePodVerticalScalingExclusiveMemory` enabled)

**TODO**: Move this logic into a `PodAdmitHandler` by adding an operation field to `lifecycle.PodAdmitAttributes`.

## Resize Retry Mechanism

### RetryPendingResizes() (line 237)

Called from multiple triggers in `kubelet.go`:
- `TriggerReasonPodsAdded` - When pods are added (line 2802)
- `TriggerReasonPodUpdated` - When pods are updated (line 2830)
- `TriggerReasonPodsRemoved` - When pods are removed (line 2987)
- `TriggerReasonPodResized` - When pod resize completes (line 3067)

**Performance Note**: Each retry calls `canAdmitPod()` which builds a fresh `NodeInfo` via `schedulerframework.NewNodeInfo(pods...)`. This is O(n×m) per retry. Caching NodeInfo would significantly benefit resize retries where the same pod set is repeatedly evaluated. See issue #132858.

## Resize Behavior

- Pending resizes sorted by: non-increasing requests, PriorityClass, QoS class, wait time
- Resizes can be deferred if resources unavailable, infeasible if fundamentally impossible
- Retries occur periodically (30s initial, then 3min intervals)
- Special handling for CPU/memory managers with static policies

## Integration Points

- **admitHandlers**: List of `lifecycle.PodAdmitHandler` including `predicateAdmitHandler`
- **getActivePods**: Function to get currently active pods
- **getAllocatedPods**: Enriches pods with their allocated resources from checkpoint

## Design Notes

- Uses checkpoint files for persistence across kubelet restarts
- Requires InPlacePodVerticalScaling feature gate
- Integrates with admission handlers for pod fit validation
- Thread-safe with mutex protection during allocation operations
- Both `canAdmitPod` and `canResizePod` use the same `allocatedPods` list for consistency
