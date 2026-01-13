# Package: tainttoleration

## Purpose
Implements taint and toleration scheduling constraints. Filters nodes with taints that pods don't tolerate and scores nodes based on toleration matches.

## Key Types

### TaintToleration
The plugin struct implementing:
- FilterPlugin, PreScorePlugin, ScorePlugin, EnqueueExtensions, SignPlugin
- **handle**: Framework handle
- **enableSchedulingQueueHint**: Feature flag for queueing hints
- **enableTaintTolerationComparisonOperators**: Feature flag for comparison operators

## Extension Points

### Filter
- Gets node taints with NoSchedule or NoExecute effects
- Checks if pod tolerates all such taints
- Returns Unschedulable if any taint is not tolerated

### PreScore
- Filters to PreferNoSchedule taints only
- Caches filtered taints for Score phase

### Score
- Counts PreferNoSchedule taints not tolerated by pod
- Lower count = higher score (prefers nodes with fewer unmatched taints)
- Normalized to [0, MaxNodeScore]

## Key Functions

- **New(ctx, obj, handle, features)**: Creates the plugin
- **SignPod(ctx, pod)**: Returns tolerations for pod signing
- **EventsToRegister()**: Returns Node/Add, Node/UpdateNodeTaint, Pod/UpdatePodToleration events
- **isSchedulableAfterNodeChange**: Queueing hint for taint changes
- **isSchedulableAfterPodTolerationChange**: Queueing hint for toleration additions

## Taint Effects
- **NoSchedule**: Hard constraint (Filter)
- **NoExecute**: Hard constraint (Filter) - also evicts running pods
- **PreferNoSchedule**: Soft constraint (Score)

## Design Pattern
- Separates hard constraints (Filter) from soft preferences (Score)
- Uses helper.DoNotScheduleTaintsFilterFunc for taint filtering
- Supports comparison operators (>=, <=) with feature gate
- Tolerations can match by key, value, effect, and operator
