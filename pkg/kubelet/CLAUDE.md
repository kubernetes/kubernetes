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

## Resize Retry Triggers

The kubelet triggers resize retries from multiple points in `kubelet.go`:
- Line 2802: `TriggerReasonPodsAdded` - When pods are added
- Line 2830: `TriggerReasonPodUpdated` - When pods are updated
- Line 2987: `TriggerReasonPodsRemoved` - When pods are removed
- Line 3067: `TriggerReasonPodResized` - When pod resize completes

## Design Notes

- The kubelet watches for pods from multiple sources (API server, file, HTTP)
- Manages container runtime via CRI (Container Runtime Interface)
- Reports node and pod status back to the API server
- Implements various managers for resources (CPU, memory, devices)
- Supports graceful node shutdown and pod eviction
- Node object caching reduces API calls during admission
- Admission handlers registered via `lifecycle.PodAdmitHandler` interface
