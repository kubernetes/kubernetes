# Package: defaultbinder

## Purpose
Implements the default pod binding plugin that assigns pods to nodes using the Kubernetes API. This is the standard implementation of the Bind extension point.

## Key Types

### DefaultBinder
The plugin struct implementing `fwk.BindPlugin`:
- **handle**: Framework handle for accessing the API client and cacher

## Key Functions

- **New(ctx, obj, handle)**: Creates a new DefaultBinder plugin instance
- **Name()**: Returns "DefaultBinder"
- **Bind(ctx, state, pod, nodeName)**: Binds a pod to the specified node

## Bind Implementation
The Bind method:
1. Creates a `v1.Binding` object with the pod's metadata and target node
2. If APICacher is available, uses it for optimized binding (batched API calls)
3. Otherwise, directly calls the Kubernetes API to bind the pod
4. Returns success or error status

## Design Pattern
- Uses the APICacher when available for efficient batched API calls
- Falls back to direct API calls when APICacher is not configured
- The binding is a create operation on the Pod's binding subresource
