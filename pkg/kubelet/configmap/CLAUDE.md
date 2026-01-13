# Package configmap

Package configmap provides ConfigMap management for the kubelet, caching ConfigMaps referenced by pods for efficient access.

## Key Types

- `Manager`: Interface for ConfigMap operations
- `simpleConfigMapManager`: Direct API server calls (no caching)
- `configMapManager`: Cached ConfigMap management with configurable freshness

## Manager Interface

- `GetConfigMap(namespace, name)`: Retrieves a ConfigMap
- `RegisterPod(pod)`: Registers all ConfigMaps referenced by a pod
- `UnregisterPod(pod)`: Unregisters ConfigMaps no longer needed

## Factory Functions

- `NewSimpleConfigMapManager(kubeClient)`: Creates direct API manager
- `NewCachingConfigMapManager(kubeClient, store)`: Creates cached manager with TTL
- `NewWatchingConfigMapManager(kubeClient, resyncInterval)`: Creates watch-based manager

## Caching Strategies

TTL-based (Caching):
- Invalidates cache on pod create/update
- Fetches from API if cache miss or expired
- Default TTL: 1 minute

Watch-based (Watching):
- Maintains cache via watch stream
- Reflects changes immediately
- More API-server friendly for large deployments

## Helper Functions

- `getConfigMapNames(pod)`: Extracts all ConfigMap names referenced by a pod

## Design Notes

- RegisterPod/UnregisterPod must be efficient (non-blocking)
- Uses manager.Manager internally for generic caching logic
- Supports volumes, envFrom, and env value references
