# DefaultBinder Plugin (`pkg/scheduler/framework/plugins/defaultbinder`)

This guide provides an architectural overview, interface implementations, binding execution pathways, asynchronous API cacher integration, error handling, and testing strategies for the `DefaultBinder` plugin in `pkg/scheduler/framework/plugins/defaultbinder`.

---

## 1. High-Level Purpose & Scope

The `DefaultBinder` plugin is the terminal extension point implementation of the Kubernetes scheduler framework's scheduling cycle. It executes the official binding operation that assigns an unscheduled pod to a selected target node by persisting a `v1.Binding` resource to the Kubernetes API server.

### Core Responsibilities:
1. **Pod-to-Node Binding**: Submits the `v1.Binding` API request associating `pod.Namespace`, `pod.Name`, and `pod.UID` with `Target: v1.ObjectReference{Kind: "Node", Name: nodeName}`.
2. **Dual Execution Pipeline**:
   - **Direct Client-Go Call**: Invokes `util.BindPod` using `b.handle.ClientSet()` when asynchronous caching is not enabled.
   - **Asynchronous / Pipelined API Cacher**: Dispatches the binding through `b.handle.APICacher().BindPod(binding)` and awaits completion via `WaitOnFinish(ctx, onFinish)` when pipelined scheduling is active.
3. **Status & Error Conversion**: Wraps binding and network errors into `fwk.Status` via `fwk.AsStatus(err)`.

---

## 2. Package Architecture & File Map

```
pkg/scheduler/framework/plugins/defaultbinder/
├── default_binder.go       # Plugin definition, Bind implementation, API cacher dispatch
├── default_binder_test.go  # Table-driven unit tests for sync and async binding flows
└── AGENTS.md               # This agent documentation
```

---

## 3. Extension Point Implementations

`DefaultBinder` implements the `fwk.BindPlugin` interface.

```
                    ┌───────────────────────────────┐
                    │      Schedule & Reserve       │
                    │   (Target Node Selected)      │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │       DefaultBinder.Bind      │
                    │   Constructs v1.Binding       │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
        [ APICacher != nil ]             [ APICacher == nil ]
                    │                               │
                    ▼                               ▼
    ┌───────────────────────────────┐   ┌───────────────────────────────┐
    │ APICacher.BindPod(binding)    │   │ util.BindPod(ctx, clientSet)  │
    │ => returns onFinish channel   │   │ => POST /api/v1/namespaces/   │
    │ APICacher.WaitOnFinish(ctx)   │   │    {ns}/pods/{name}/binding   │
    └───────────────┬───────────────┘   └───────────────┬───────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    │
                         ┌──────────┴──────────┐
                         │                     │
                    [ Success ]            [ Error ]
                         │                     │
                         ▼                     ▼
                  ┌──────────────┐      ┌─────────────────────────┐
                  │  Return nil  │      │ Return fwk.AsStatus(err)│
                  └──────────────┘      └─────────────────────────┘
```

### 3.1. `Bind` (`BindPlugin`)
- **Signature**: `Bind(ctx context.Context, state fwk.CycleState, p *v1.Pod, nodeName string) *fwk.Status`
- **Execution Workflow**:
  1. Instantiates a `*v1.Binding` object:
     ```go
     binding := &v1.Binding{
         ObjectMeta: metav1.ObjectMeta{Namespace: p.Namespace, Name: p.Name, UID: p.UID},
         Target:     v1.ObjectReference{Kind: "Node", Name: nodeName},
     }
     ```
  2. Evaluates whether the framework handle provides an `APICacher`:
     - **Async APICacher Path**:
       ```go
       onFinish, err := b.handle.APICacher().BindPod(binding)
       if err != nil {
           return fwk.AsStatus(err)
       }
       err = b.handle.APICacher().WaitOnFinish(ctx, onFinish)
       if err != nil {
           return fwk.AsStatus(err)
       }
       return nil
       ```
     - **Synchronous ClientSet Path**:
       ```go
       logger.V(3).Info("Attempting to bind pod to node", "pod", klog.KObj(p), "node", klog.KRef("", nodeName))
       err := util.BindPod(ctx, b.handle.ClientSet(), binding)
       if err != nil {
           return fwk.AsStatus(err)
       }
       return nil
       ```

---

## 4. Key Constants & Status Invariants

| Identifier | Value / Type | Purpose |
| :--- | :--- | :--- |
| `Name` | `names.DefaultBinder` (`"DefaultBinder"`) | Registered plugin name in scheduler profile. |
| Target ObjectReference | `v1.ObjectReference{Kind: "Node", Name: nodeName}` | Defines the bound host node in the `v1.Binding` payload. |
| Return Status | `nil` (Success) or `fwk.AsStatus(err)` (Error) | Signals whether the binding was successfully committed. |

---

## 5. Error Handling & Edge Cases

1. **Context Cancellation & Timeouts**: When the scheduling context is cancelled while waiting on `WaitOnFinish` or during client HTTP requests, `ctx.Err()` is propagated and converted into an error status.
2. **API Server Rejection**: If the API server returns a 409 Conflict (e.g. pod was deleted or bound concurrently) or 403 Forbidden, `fwk.AsStatus` translates the client-go error into a scheduler status.
3. **UID Verification**: Including `p.UID` in `binding.ObjectMeta.UID` ensures that the binding request fails if the pod was deleted and recreated with the same name before the binding completed.

---

## 6. Test Fixtures & Unit Testing

`default_binder_test.go` exercises both synchronous and asynchronous binding flows using fake client reactors:

- **Reactor Interception**: Intercepts `create pods` actions where `action.GetSubresource() == "binding"` to extract and assert the generated `*v1.Binding`.
- **Async APICacher Setup**: Initializes `apidispatcher.New(client, 16, apicalls.Relevances)` and `apicache.New(nil, cache)` to verify the `WaitOnFinish` synchronization lifecycle.
- **Error Injection**: Injects synthetic errors into the reactor to verify proper `fwk.AsStatus(err)` status conversion.
