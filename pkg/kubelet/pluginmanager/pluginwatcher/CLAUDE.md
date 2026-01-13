# Package: pluginwatcher

Watches filesystem for plugin socket registration and populates desired state of world.

## Key Types

- **Watcher**: Monitors a directory for Unix domain socket creation/deletion.

## Key Functions

- `NewWatcher(sockDir, dsw)`: Creates watcher for the specified directory.
- `Start(ctx, stopCh)`: Starts watching, initializes directory, and discovers existing plugins.

## Behavior

### Socket Discovery
1. Creates watched directory if it doesn't exist
2. Traverses directory tree recursively
3. Adds fsnotify watchers to all directories
4. Registers existing Unix domain sockets

### Event Handling
- **Create**: If socket, adds to DesiredStateOfWorld; if directory, adds watcher
- **Delete**: Removes socket from DesiredStateOfWorld

### File Filtering
- Ignores files starting with '.' (hidden files)
- Only processes Unix domain sockets
- Watches subdirectories recursively

## Integration

```
Filesystem Events --> Watcher --> DesiredStateOfWorld --> Reconciler
```

## Platform Support

- Default implementation (plugin_watcher.go)
- Windows-specific handling (plugin_watcher_windows.go)
- Other platforms (plugin_watcher_others.go)

## Example Plugin

Package includes example_plugin.go and example_handler.go for testing.
