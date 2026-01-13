# Package: cache

State caches and handler interfaces for the plugin manager.

## Key Interfaces

- **PluginHandler**: Interface that plugin consumers must implement:
  - `ValidatePlugin(name, endpoint, versions)`: Validates plugin information
  - `RegisterPlugin(name, endpoint, versions, timeout)`: Registers the plugin
  - `DeRegisterPlugin(name, endpoint)`: Deregisters the plugin

- **ActualStateOfWorld**: Tracks currently registered plugins
- **DesiredStateOfWorld**: Tracks plugins that should be registered

## Plugin Handler State Machine

```
Socket Created --> Validate --> Register --> (ReRegistration or DeRegister)
                     |             |
                   Error         Error
                     v             v
                    Out           Out
```

## Key Types

- **PluginInfo**: Contains socket path, timestamp, handler, and plugin name.

## State Cache Operations

### ActualStateOfWorld
- `AddPlugin()`: Add a registered plugin
- `RemovePlugin()`: Remove a plugin
- `GetRegisteredPlugins()`: Get all registered plugins
- `PluginExistsWithCorrectTimestamp()`: Check if plugin exists with matching timestamp

### DesiredStateOfWorld
- `AddOrUpdatePlugin()`: Add or update plugin with current timestamp
- `RemovePlugin()`: Remove plugin
- `GetPluginsToRegister()`: Get plugins needing registration

## Design Notes

- ReRegistration happens when a new socket is created with the same plugin name
- DeRegistration triggered only by deletion of the new socket (not old one)
- Timestamps used to detect plugin updates
