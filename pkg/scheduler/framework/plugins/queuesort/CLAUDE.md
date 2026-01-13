# Package: queuesort

## Purpose
Implements the default queue sorting plugin that determines the order in which pods are scheduled. Sorts pods by priority, with timestamp as tiebreaker.

## Key Types

### PrioritySort
The plugin struct implementing `fwk.QueueSortPlugin`:
- Stateless - no configuration needed

## Key Functions

- **New(ctx, obj, handle)**: Creates a new PrioritySort plugin
- **Name()**: Returns "PrioritySort"
- **Less(pInfo1, pInfo2)**: Compares two queued pods for ordering

## Sorting Logic
```go
func (pl *PrioritySort) Less(pInfo1, pInfo2 fwk.QueuedPodInfo) bool {
    p1 := corev1helpers.PodPriority(pInfo1.GetPodInfo().GetPod())
    p2 := corev1helpers.PodPriority(pInfo2.GetPodInfo().GetPod())
    return (p1 > p2) || (p1 == p2 && pInfo1.GetTimestamp().Before(pInfo2.GetTimestamp()))
}
```

1. Higher priority pods are scheduled first
2. For equal priority, older pods (earlier timestamp) are scheduled first

## Design Pattern
- Simple comparison-based sorting
- Uses pod.spec.priority (from PriorityClass)
- FIFO within same priority level
- The only built-in QueueSort plugin
- Can be replaced by custom plugins for different ordering strategies
