# Scheduler Plugin Testing Utilities (`pkg/scheduler/framework/plugins/testing`)

This guide provides an architectural overview, test harness mechanics, informer synchronization lifecycle, and usage patterns for the shared testing utilities in `pkg/scheduler/framework/plugins/testing`.

---

## 1. High-Level Purpose & Scope

The `testing` package provides standardized test harness factories for creating isolated scheduler framework instances during unit testing. It eliminates boilerplate code when instantiating plugins that require a `fwk.Handle`, `fwk.SharedLister`, or active client-go `informers.SharedInformerFactory`.

### Core Responsibilities:
1. **Lightweight Framework Setup**: Instantiates `fwk.Handle` with a mock or snapshot `SharedLister` via `SetupPlugin`.
2. **Full Informer Harness Setup**: Instantiates `fwk.Handle` backed by `fake.NewClientset`, starts informer routines, creates default namespaces, and blocks until cache synchronization completes via `SetupPluginWithInformers`.
3. **Deterministic Test Isolation**: Prevents shared state leakage between unit tests by encapsulating all clients and listers within the test's `context.Context`.

---

## 2. Package Architecture & File Map

```
pkg/scheduler/framework/plugins/testing/
├── testing.go       # SetupPlugin and SetupPluginWithInformers test harness helpers
└── AGENTS.md        # This agent documentation
```

---

## 3. Harness Factory Functions

```
                             ┌───────────────────────────────────────────────┐
                             │       SetupPluginWithInformers                │
                             │ (ctx, tb, pf, config, sharedLister, objs)     │
                             └───────────────────────┬───────────────────────┘
                                                     │
                                                     ▼
                             ┌───────────────────────────────────────────────┐
                             │ 1. Inject default namespace ("")              │
                             │ 2. Instantiate fake.NewClientset(objs...)     │
                             │ 3. Create informers.SharedInformerFactory     │
                             │ 4. Build framework handle (fh)                │
                             │ 5. Execute PluginFactory(ctx, config, fh)     │
                             │ 6. informerFactory.Start(ctx.Done())          │
                             │ 7. informerFactory.WaitForCacheSync(...)      │
                             └───────────────────────┬───────────────────────┘
                                                     │
                                                     ▼
                             ┌───────────────────────────────────────────────┐
                             │ Return initialized fwk.Plugin                 │
                             └───────────────────────────────────────────────┘
```

### 3.1. `SetupPlugin`
- **Signature**:
  ```go
  func SetupPlugin(
      ctx context.Context,
      tb testing.TB,
      pf frameworkruntime.PluginFactory,
      config runtime.Object,
      sharedLister fwk.SharedLister,
  ) fwk.Plugin
  ```
- **Use Case**: For plugins that only read from `SnapshotSharedLister` (e.g. node capacity, cached node info, pod group states) and do not need live informer event streams.

### 3.2. `SetupPluginWithInformers`
- **Signature**:
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
- **Execution Workflow**:
  1. Appends an empty namespace (`&v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: ""}}`) to `objs` because most scheduler unit tests generate pods without explicit namespaces.
  2. Creates a `fake.Clientset` populated with `objs`.
  3. Creates an informer factory and registers it onto the framework handle via `frameworkruntime.WithInformerFactory`.
  4. Calls the plugin factory `pf(ctx, config, fh)`.
  5. Starts informers in background goroutines tied to `ctx.Done()`.
  6. Blocks on `WaitForCacheSync(ctx.Done())` to ensure all listers are fully populated before test assertions run.

---

## 4. Usage Patterns & Best Practices

```go
func TestMyPlugin(t *testing.T) {
    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()

    nodes := []*v1.Node{st.MakeNode().Name("node-1").Obj()}
    snapshot := cache.NewSnapshot(nil, nodes)

    plugin := testinghelper.SetupPlugin(
        ctx,
        t,
        myplugin.New,
        &config.MyPluginArgs{},
        snapshot,
    )
    
    // Execute plugin methods (PreFilter, Filter, Score)
}
```
