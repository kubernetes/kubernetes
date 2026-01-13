# Package: helper

## Purpose
Provides utility functions shared across multiple scheduler plugins. Contains helpers for score normalization, selector computation, taint filtering, and score shaping.

## Key Functions

### Score Normalization
- **DefaultNormalizeScore(maxPriority, reverse, scores)**: Normalizes scores from [0, max] to [0, maxPriority]. Optionally reverses scores (for preferring lower values).

### Selector Computation
- **DefaultSelector(pod, serviceLister, rcLister, rsLister, ssLister)**: Computes a label selector from the pod's owning controllers (Services, ReplicationControllers, ReplicaSets, StatefulSets). Used by InterPodAffinity and PodTopologySpread.

- **GetPodServices(serviceLister, pod)**: Returns all Services whose selector matches the pod's labels.

### Taint Filtering
- **DoNotScheduleTaintsFilterFunc()**: Returns a filter function that selects only NoSchedule and NoExecute taints. Used by TaintToleration plugin.

### Score Shaping
- **FunctionShape**: Slice of FunctionShapePoint defining a piecewise linear function
- **FunctionShapePoint**: Point with Utilization and Score values
- **BuildBrokenLinearFunction(shape)**: Creates a piecewise linear scoring function from shape points. Used by NodeResourcesFit for RequestedToCapacityRatio scoring.

## Key Types

### FunctionShape / FunctionShapePoint
Define scoring curves for resource utilization:
```go
type FunctionShapePoint struct {
    Utilization int64  // x-axis: resource utilization
    Score       int64  // y-axis: resulting score
}
```

## Design Pattern
- Stateless utility functions
- Shared by multiple plugins to avoid code duplication
- Composable functions (e.g., filter functions, shape builders)
