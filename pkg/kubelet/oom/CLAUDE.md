# Package: oom

Watches for Out-Of-Memory (OOM) events and records them as Kubernetes events.

## Key Types

- **Watcher**: Interface for OOM watchers with a `Start(ctx, ref)` method.
- **realWatcher**: Linux implementation using cadvisor's oomparser.
- **streamer**: Internal interface for OOM event streaming (implemented by oomparser.OomParser).

## Key Functions

- `NewWatcher(recorder)`: Creates a new OOM watcher backed by cadvisor's oomparser. Returns nil for fake recorders (test environments).
- `Start(ctx, ref)`: Starts watching for system OOM events and records them as Kubernetes Warning events.

## Behavior

- Listens for OOM events from cadvisor's oomparser
- Only records events for the root container ("/") - system-level OOMs
- Records "SystemOOM" events with victim process name and PID
- Runs as a background goroutine

## Platform Support

- **Linux**: Full implementation using cadvisor's oomparser
- **Other platforms**: Unsupported (oom_watcher_unsupported.go)

## Event Format

```
Type: Warning
Reason: SystemOOM
Message: System OOM encountered, victim process: <name>, pid: <pid>
```
