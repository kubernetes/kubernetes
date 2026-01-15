# Package: kubelet

## Purpose
This is the main package for the Kubernetes Kubelet - the primary node agent that runs on each node in the cluster. The Kubelet ensures that containers are running in pods and manages the complete lifecycle of pods on a node.

## Key Components

The kubelet package is organized into many subpackages:
- **allocation/**: Pod resource allocation tracking for in-place vertical scaling
- **apis/config/**: Kubelet configuration API types and versioning
- **cadvisor/**: Container metrics collection
- **cm/**: Container manager (cgroups, CPU manager, memory manager)
- **eviction/**: Pod eviction based on resource pressure
- **kuberuntime/**: CRI runtime integration
- **lifecycle/**: Pod lifecycle handlers and admission
- **prober/**: Container liveness, readiness, and startup probes
- **status/**: Pod status management and reporting

## Node Caching (kubelet_nodecache.go)

The kubelet caches the Node object to avoid repeated API calls during admission:

### GetCachedNode(ctx, useCache)

```go
func (kl *Kubelet) GetCachedNode(ctx context.Context, useCache bool) (*v1.Node, error)
```

- **useCache=true**: Returns cached node, updates if informer has newer ResourceVersion
- **useCache=false**: Forces synchronous API call to get fresh node data
- **Fallback**: Returns `kl.cachedNode` if informer fails, or synthetic initial node

### Caching Strategy

1. Compares `node.ResourceVersion` to detect newer versions
2. Updates `kl.cachedNode` only when informer node is newer
3. Used by `predicateAdmitHandler` for pod admission decisions

**Note**: Only the Node object is cached. NodeInfo (aggregated pod resources) is reconstructed on every admission. See issue #132858 for NodeInfo caching improvements.

## Pod Lifecycle Event Handlers

The kubelet processes pod events through handlers in `kubelet.go`. These are called from `syncLoopIteration()` when events arrive on the config channel.

### HandlePodAdditions (line 2717)

Called when pods are added from config sources (API server, file, HTTP):

| Step | Line | Operation |
|------|------|-----------|
| 1 | 2719 | Sort pods by creation time |
| 2 | 2726 | `podManager.AddPod(pod)` - add to pod manager |
| 3 | 2756 | `allocationManager.AddPod()` - **admission check** |
| 4 | 2757-2763 | If rejected, call `rejectPod()` and continue |
| 5 | 2776-2781 | `podWorkers.UpdatePod()` - queue async sync |
| 6 | 2789 | `RetryPendingResizes(TriggerReasonPodsAdded)` |

### HandlePodUpdates (line 2796)

Called when pod specs change:

| Step | Line | Operation |
|------|------|-----------|
| 1 | 2799 | Get old pod from pod manager |
| 2 | 2800 | `podManager.UpdatePod(pod)` |
| 3 | 2811 | `recordResizeOperations()` - detect resource changes |
| 4 | 2814 | If resize, `PushPendingResize()` |
| 5 | 2817 | `RetryPendingResizes(TriggerReasonPodUpdated)` |
| 6 | 2847-2852 | `podWorkers.UpdatePod()` - queue async sync |

### HandlePodRemoves (line 2944)

Called when pods are deleted:

| Step | Line | Operation |
|------|------|-----------|
| 1 | 2947 | `podCertificateManager.ForgetPod()` |
| 2 | 2948 | `podManager.RemovePod(pod)` |
| 3 | 2949 | `allocationManager.RemovePod()` |
| 4 | 2968 | `deletePod()` - initiate async termination |
| 5 | 2974 | `RetryPendingResizes(TriggerReasonPodsRemoved)` |

## NewMainKubelet Initialization (line 422)

Key component creation sequence in `NewMainKubelet()`:

| Line | Component | Dependencies |
|------|-----------|--------------|
| 689 | `podManager` | None |
| 691 | `statusManager` | kubeClient, podManager |
| 692-702 | `allocationManager` | statusManager, GetActivePods callback |
| 728-736 | `podWorkers` | allocationManager |
| 1042-1080 | Admission handlers created | Various |
| 1065 | `predicateAdmitHandler` | GetCachedNode, containerManager |
| 1112 | `AddPodAdmitHandlers()` | All handlers registered |

### Admission Handler Registration (lines 1042-1112)

Handlers are created and added in order:
1. `evictionAdmitHandler` (line 1043)
2. `sysctlsAllowlist` (line 1052)
3. Container manager handler (line 1062)
4. `predicateAdmitHandler` (line 1065) - core resource admission
5. `appArmorAdmitHandler` (line 1074)
6. `podFeaturesAdmitHandler` (line 1077)
7. `declaredFeaturesAdmitHandler` (line 1080)
8. `shutdownManager` (line 1111)

All handlers registered via `allocationManager.AddPodAdmitHandlers(handlers)` at line 1112.

## Resize Retry Triggers

The kubelet triggers resize retries from multiple points in `kubelet.go`:
- Line 2789: `TriggerReasonPodsAdded` - When pods are added
- Line 2817: `TriggerReasonPodUpdated` - When pods are updated
- Line 2974: `TriggerReasonPodsRemoved` - When pods are removed
- Line 3054: `TriggerReasonPodResized` - When pod resize completes

## Design Notes

- The kubelet watches for pods from multiple sources (API server, file, HTTP)
- Manages container runtime via CRI (Container Runtime Interface)
- Reports node and pod status back to the API server
- Implements various managers for resources (CPU, memory, devices)
- Supports graceful node shutdown and pod eviction
- Node object caching reduces API calls during admission
- Admission handlers registered via `lifecycle.PodAdmitHandler` interface
