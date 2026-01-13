# Package: nodename

## Purpose
Implements a simple filter plugin that checks if a pod specifies a particular node name and ensures it only schedules to that node.

## Key Types

### NodeName
The plugin struct implementing:
- FilterPlugin, EnqueueExtensions, SignPlugin
- **enableSchedulingQueueHint**: Feature flag for queueing hints

## Extension Points

### Filter
- If pod.spec.nodeName is empty, allows all nodes (returns nil)
- If pod.spec.nodeName is set, only allows the matching node
- Returns UnschedulableAndUnresolvable for non-matching nodes

## Key Functions

- **New(ctx, obj, handle, features)**: Creates the plugin
- **Fits(pod, nodeInfo)**: Static function checking if pod fits node
- **SignPod(ctx, pod)**: Returns nodeName for pod signing
- **EventsToRegister()**: Returns Node/Add events

## Fits Logic
```go
func Fits(pod *v1.Pod, nodeInfo fwk.NodeInfo) bool {
    return len(pod.Spec.NodeName) == 0 || pod.Spec.NodeName == nodeInfo.Node().Name
}
```

## Design Pattern
- Simplest filter plugin - direct string comparison
- No scoring component (exact match or no match)
- Returns UnschedulableAndUnresolvable because the condition can't be resolved by cluster changes (except node addition)
- Used for pods that must run on specific nodes (e.g., DaemonSet pods after node selection)
