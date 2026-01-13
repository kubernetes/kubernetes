# Package: schedulinggates

## Purpose
Implements a PreEnqueue plugin that blocks pods with scheduling gates from entering the scheduling queue until all gates are removed.

## Key Types

### SchedulingGates
The plugin struct implementing:
- PreEnqueuePlugin, EnqueueExtensions
- **enableSchedulingQueueHint**: Feature flag for queueing hints

## Extension Points

### PreEnqueue
- Checks if pod.spec.schedulingGates is non-empty
- Returns nil (allow) if no gates
- Returns UnschedulableAndUnresolvable with gate names if gates exist

## Key Functions

- **New(ctx, obj, handle, features)**: Creates the plugin
- **PreEnqueue(ctx, pod)**: Checks for scheduling gates
- **EventsToRegister()**: Returns Pod/UpdatePodSchedulingGatesEliminated event
- **isSchedulableAfterUpdatePodSchedulingGatesEliminated**: Queueing hint for gate removal

## Scheduling Gates Usage
Pods can specify scheduling gates to delay scheduling:
```yaml
spec:
  schedulingGates:
  - name: "example.com/wait-for-config"
  - name: "example.com/wait-for-secrets"
```

## Gate Removal
External controllers remove gates when ready:
```yaml
spec:
  schedulingGates: []  # All gates removed
```

## Design Pattern
- Pre-scheduling hold mechanism for external coordination
- Used by controllers that need to prepare resources before pod scheduling
- Gates are removed (not added) - list only shrinks
- Pod enters queue only when all gates are removed
- Common use case: cluster autoscaler provisioning, secrets injection
