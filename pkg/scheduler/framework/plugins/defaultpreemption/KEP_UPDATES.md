# Proposed KEP-5710 Updates

## PodGroupPostFilter Interface Adjustment

This document captures the rationale for a proposed update to KEP-5710 regarding the `PodGroupPostFilterPlugin` interface definition.

### Current KEP Definition
The original KEP-5710 defines the interface as taking the API type `*v1alpha3.PodGroup` directly:

```go
type PodGroupPostFilterPlugin interface {
    Plugin
    PodGroupPostFilter(ctx context.Context, pg *v1alpha3.PodGroup, pods []*v1.Pod, pgSchedulingFunc PodGroupSchedulingFunc) (*PodGroupPostFilterResult, *Status)
}
```

### Proposed Update (Implemented)
During implementation, the interface was updated to use `*framework.PodGroupInfo` instead:

```go
type PodGroupPostFilterPlugin interface {
    Plugin
    PodGroupPostFilter(ctx context.Context, pgInfo PodGroupInfo, pods []*v1.Pod, pgSchedulingFunc PodGroupSchedulingFunc) (*PodGroupPostFilterResult, *Status)
}
```

### Rationale
`PodGroupInfo` is the standard external interface wrapping the `PodGroup` used throughout the gang scheduling framework. By passing the interface `PodGroupInfo` rather than the raw `*v1alpha3.PodGroup`:
1. It aligns the `PodGroupPostFilter` extension point with existing internal framework data structures.
2. It prevents plugins from needing to re-fetch the PodGroup object or maintain their own internal caching.
3. It guarantees that the plugin evaluates exactly the same state wrapper that the scheduler algorithm was operating on during the failed scheduling cycle.
