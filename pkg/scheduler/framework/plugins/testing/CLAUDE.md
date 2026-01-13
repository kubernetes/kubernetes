# Package: testing

## Purpose
Provides test utilities for setting up scheduler plugins in unit tests. Simplifies plugin instantiation with mock framework handles and informers.

## Key Functions

### SetupPluginWithInformers
```go
func SetupPluginWithInformers(
    ctx context.Context,
    tb testing.TB,
    pf frameworkruntime.PluginFactory,
    config runtime.Object,
    sharedLister fwk.SharedLister,
    objs []runtime.Object,
) fwk.Plugin
```
- Creates a plugin with a full informer factory
- Automatically creates an empty namespace (most tests use empty namespace)
- Starts informers and waits for cache sync
- Fails test on any error

### SetupPlugin
```go
func SetupPlugin(
    ctx context.Context,
    tb testing.TB,
    pf frameworkruntime.PluginFactory,
    config runtime.Object,
    sharedLister fwk.SharedLister,
) fwk.Plugin
```
- Creates a plugin with just a shared lister (no informers)
- Simpler setup for plugins that don't need informers
- Fails test on any error

## Usage Example
```go
func TestMyPlugin(t *testing.T) {
    ctx := context.Background()
    plugin := testing.SetupPluginWithInformers(
        ctx, t,
        myplugin.New,
        &config.MyPluginArgs{},
        snapshot,
        []runtime.Object{node1, pod1},
    )
    // Use plugin...
}
```

## Design Pattern
- Reduces boilerplate in plugin unit tests
- Uses fake clientset for API interactions
- Automatically handles informer lifecycle
- Fails fast on setup errors (tb.Fatal)
