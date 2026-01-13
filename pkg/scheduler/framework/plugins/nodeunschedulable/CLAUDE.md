# Package: nodeunschedulable

## Purpose
Implements a filter plugin that prevents scheduling on nodes marked as unschedulable (node.spec.unschedulable=true), unless the pod tolerates the unschedulable taint.

## Key Types

### NodeUnschedulable
The plugin struct implementing:
- FilterPlugin, EnqueueExtensions, SignPlugin
- **enableSchedulingQueueHint**: Feature flag for queueing hints

## Extension Points

### Filter
- Returns nil (allow) if node.spec.unschedulable is false
- Checks if pod tolerates the `node.kubernetes.io/unschedulable:NoSchedule` taint
- Returns UnschedulableAndUnresolvable if node is unschedulable and pod doesn't tolerate it

## Key Functions

- **New(ctx, obj, handle, features)**: Creates the plugin
- **SignPod(ctx, pod)**: Returns tolerations for pod signing
- **EventsToRegister()**: Returns Node/Add, Node/UpdateNodeTaint, and Pod/UpdatePodToleration events
- **isSchedulableAfterNodeChange**: Queueing hint for node schedulability changes
- **isSchedulableAfterPodTolerationChange**: Queueing hint for toleration additions

## Toleration Check
A pod can schedule on an unschedulable node if it tolerates:
```yaml
- key: node.kubernetes.io/unschedulable
  effect: NoSchedule
```

## Design Pattern
- Simple boolean check with toleration override
- Used by the node controller to cordon nodes
- Allows system pods (with appropriate tolerations) to schedule on cordoned nodes
- Comparison operators support controlled by feature flag
