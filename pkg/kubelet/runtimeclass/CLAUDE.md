# Package: runtimeclass

Caches RuntimeClass API objects and provides handler lookup for the kubelet.

## Key Types

- **Manager**: Caches RuntimeClass objects using a shared informer.

## Key Functions

- `NewManager(client)`: Creates a new RuntimeClass manager with an informer factory.
- `Start(stopCh)`: Starts syncing the RuntimeClass cache with the API server.
- `WaitForCacheSync(stopCh)`: Waits for the informer cache to sync (exposed for testing).
- `LookupRuntimeHandler(runtimeClassName)`: Returns the handler string for a RuntimeClass name.
  - Returns "" for nil/empty class name (default runtime)
  - Returns errors.NotFound if RuntimeClass doesn't exist

## Design Notes

- Uses SharedInformerFactory with no resync period (resyncPeriod = 0)
- RuntimeClass handler maps to CRI runtime handler name
- Nil/empty RuntimeClassName always resolves to empty handler (default runtime)
- Cache-based lookups avoid API server calls for pod admission
