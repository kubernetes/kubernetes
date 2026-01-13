# Package: config

This package implements configuration watchers that monitor Kubernetes API objects (Services, EndpointSlices, Nodes) and notify registered handlers of changes.

## Key Types

- `ServiceConfig` - Watches Service objects and notifies ServiceHandler implementations
- `EndpointSliceConfig` - Watches EndpointSlice objects and notifies EndpointSliceHandler
- `NodeConfig` - Watches Node objects and notifies NodeHandler implementations
- `ServiceHandler` - Interface for receiving Service add/update/delete/sync events
- `EndpointSliceHandler` - Interface for receiving EndpointSlice events
- `NodeHandler` - Interface for receiving Node events

## Key Functions

- `NewServiceConfig()` - Creates a new ServiceConfig with the given informer
- `NewEndpointSliceConfig()` - Creates a new EndpointSliceConfig
- `NewNodeConfig()` - Creates a new NodeConfig
- `RegisterEventHandler()` - Registers a handler to receive object change events

## Design Notes

- Uses Kubernetes SharedInformer pattern for efficient API watching
- Handlers receive full object state (not deltas) for simplicity
- Supports multiple registered handlers per config object
- Thread-safe via mutex protection of handler registration
- Calls OnSynced() once initial list is complete
