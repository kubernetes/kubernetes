# Package: lifecycle

Provides pod lifecycle management including admission handlers, sync handlers, and container lifecycle hook execution.

## Key Interfaces

- **PodAdmitHandler**: Evaluates if a pod can be admitted to the node.
- **PodAdmitTarget**: Maintains list of PodAdmitHandlers.
- **PodSyncLoopHandler**: Invoked during each sync loop to check if pod needs syncing.
- **PodSyncHandler**: Invoked during sync to determine if pod should be evicted.
- **PodLifecycleTarget**: Groups all lifecycle interfaces together.

## Key Types

- **PodAdmitAttributes**: Context for admission decisions (pod + other pods).
- **PodAdmitResult**: Result with Admit bool, Reason, and Message.
- **ShouldEvictResponse**: Result for eviction decisions.
- **handlerRunner**: Executes container lifecycle hooks (exec, HTTP, sleep).

## Admission Handlers

- **predicateAdmitHandler**: Core resource-based admission (see detailed section below).
- **appArmorAdmitHandler**: Validates AppArmor profiles for pods.
- **podFeaturesAdmitHandler**: Checks pod-level resource support.
- **declaredFeaturesAdmitHandler**: Validates pod feature requirements against node capabilities.

## predicateAdmitHandler (predicate.go)

The core admission handler that checks if a pod fits on the node based on resources, taints, and affinity.

### Structure
```go
type predicateAdmitHandler struct {
    getNodeAnyWayFunc        getNodeAnyWayFuncType
    pluginResourceUpdateFunc pluginResourceUpdateFuncType
    admissionFailureHandler  AdmissionFailureHandler
}
```

### Admit() Flow

1. **Get Node**: Calls `getNodeAnyWayFunc(ctx, true)` to get cached node
2. **OS Checks**: Validates pod OS selector and OS field match node
3. **SupplementalGroupsPolicy Check**: Validates node supports required policy
4. **Build NodeInfo**: `schedulerframework.NewNodeInfo(pods...)` - iterates all pods/containers
5. **Plugin Resources**: Calls `pluginResourceUpdateFunc` to sanitize allocatable resources
6. **Extended Resources**: Removes missing extended resources from pod requirements
7. **General Filter**: Runs resource fit, taint toleration, and affinity checks

### Performance Note

`NewNodeInfo(pods...)` is **O(n×m)** where n=pods, m=containers because it iterates through every resource in every container 3 times (desired, actual, allocated). This is called on every admission and resize retry. See issue #132858 for caching improvements.

### Affinity Fallback

If admission fails only due to node affinity (stale labels), the handler:
1. Fetches node synchronously with `getNodeAnyWayFunc(ctx, false)` (bypasses cache)
2. Updates NodeInfo with fresh node
3. Retries the general filter

### generalFilter() Checks

- **AdmissionCheck**: Resource requests vs allocatable (CPU, memory, ephemeral-storage, pods)
- **Taint Toleration**: NoExecute taints must be tolerated (except static pods)
- **Node Affinity**: Pod node selector must match node labels

## Key Functions

- `NewPredicateAdmitHandler()`: Creates the core resource admission handler.
- `NewHandlerRunner()`: Creates lifecycle hook executor.
- `Run()`: Executes lifecycle hooks (Exec, HTTPGet, or Sleep actions).
- `NewAppArmorAdmitHandler()`: Creates AppArmor validator.
- `NewPodFeaturesAdmitHandler()`: Creates pod features checker.
- `NewDeclaredFeaturesAdmitHandler()`: Creates node feature requirements checker.
- `removeMissingExtendedResources()`: Filters out extended resources not available on node.

## Error Types

- **InsufficientResourceError**: Resource limit exceeded (CPU, memory, storage, pods).
- **PredicateFailureError**: Generic predicate failure with name and description.

## Admission Failure Reasons

| Constant | Meaning |
|----------|---------|
| `OutOfCPU` | Insufficient CPU |
| `OutOfMemory` | Insufficient memory |
| `OutOfEphemeralStorage` | Insufficient ephemeral storage |
| `OutOfPods` | Pod limit reached |
| `PodOSSelectorNodeLabelDoesNotMatch` | OS label mismatch |
| `PodOSNotSupported` | OS field mismatch |
| `InvalidNodeInfo` | Cannot get node info |

## Design Notes

- Lifecycle hooks support exec commands, HTTP requests, and sleep actions
- HTTP hooks fall back from HTTPS to HTTP on certificate errors
- Sleep action gated by PodLifecycleSleepAction feature
- Handler lists are append-only slices
- Node caching uses ResourceVersion comparison for freshness
- Plugin resource update can modify NodeInfo.Allocatable for device plugins
