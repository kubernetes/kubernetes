# Package: secret

Manages Kubernetes secrets for pods, with caching and watch-based refresh.

## Key Types

- **Manager**: Interface for secret retrieval and pod registration.
- **simpleSecretManager**: Direct API calls, no caching.
- **secretManager**: Wrapper using manager.Manager for caching.

## Key Functions

- `NewSimpleSecretManager(kubeClient)`: Creates manager that fetches directly from API server.
- `NewCachingSecretManager(kubeClient, getTTL)`: Creates TTL-based caching manager.
  - Invalidates cache on pod create/update
  - Refreshes stale entries on GetSecret()
  - Default TTL: 1 minute

- `NewWatchingSecretManager(kubeClient, resyncInterval)`: Creates watch-based manager.
  - Starts individual watches for each referenced secret
  - Returns values from local cache propagated via watches
  - Efficient for frequently accessed secrets

## Interface Methods

- `GetSecret(namespace, name)`: Returns secret, fetching if needed.
- `RegisterPod(pod)`: Registers secrets referenced by pod.
- `UnregisterPod(pod)`: Unregisters secrets no longer needed.

## Design Notes

- Register/UnregisterPod must be efficient (no blocking network calls)
- Uses podutil.VisitPodSecretNames to discover secret references
- Supports immutable secrets optimization (stops watching after initial fetch)
- Underlying manager.Manager handles reference counting and caching
