# Package: reconciler

Reconciles desired state of world with actual state of world by triggering plugin register/unregister operations.

## Key Types

- **Reconciler**: Interface for running reconciliation loop and adding handlers.
- **reconciler**: Implementation that periodically compares states and triggers operations.

## Key Functions

- `NewReconciler(executor, sleepDuration, dsw, asw)`: Creates reconciler with operation executor and state caches.
- `Run(stopCh)`: Starts periodic reconciliation loop.
- `AddHandler(pluginType, handler)`: Adds handler for a plugin type.

## Reconciliation Logic

For each plugin in DesiredStateOfWorld:
1. If not in ActualStateOfWorld: trigger RegisterPlugin
2. If in ActualStateOfWorld but timestamp differs: trigger re-registration

For each plugin in ActualStateOfWorld:
1. If not in DesiredStateOfWorld: trigger UnregisterPlugin

## Configuration

- **loopSleepDuration**: Time between reconciliation iterations (default 1 second)
- Uses exponential backoff for failed operations via goroutinemap

## Thread Safety

Uses sync.RWMutex to protect the handlers map during concurrent access.

## Integration

```
Reconciler --> OperationExecutor --> PluginHandler
                     |
                     v
              ActualStateOfWorld
```
