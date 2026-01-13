# Package: pluginmanager

Manages plugin registration/deregistration lifecycle using a desired-state-of-world pattern.

## Key Types

- **PluginManager**: Interface for running plugin management loops and adding handlers.
- **pluginManager**: Implementation coordinating plugin watcher, reconciler, and state caches.

## Key Functions

- `NewPluginManager(sockDir, recorder)`: Creates plugin manager watching the specified socket directory.
- `Run(ctx, sourcesReady, stopCh)`: Starts the plugin watcher and reconciler loops.
- `AddHandler(pluginType, handler)`: Registers a handler for a specific plugin type.

## Architecture

```
Plugin Watcher (populates)  -->  Desired State of World
                                         |
                                         v
                                    Reconciler (compares)
                                         |
                                         v
Operation Executor (registers)  -->  Actual State of World
```

## Components

- **desiredStateOfWorldPopulator**: Plugin watcher that monitors socket directory for plugin sockets
- **reconciler**: Compares desired vs actual state and triggers register/unregister operations
- **actualStateOfWorld**: Cache of currently registered plugins
- **desiredStateOfWorld**: Cache of plugins that should be registered

## Design Notes

- Plugins register via Unix domain sockets in the watched directory
- Reconciler runs every 1 second (loopSleepDuration)
- Supports multiple plugin types (CSI, device plugins, etc.)
- Uses operation executor to prevent concurrent operations on same socket
