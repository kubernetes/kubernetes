# Package: defaultpreemption

## Purpose
Implements the default preemption plugin that evicts lower-priority pods to make room for higher-priority pending pods. This is the standard implementation of the PostFilter extension point.

## Key Types

### DefaultPreemption
The plugin struct implementing `fwk.PostFilterPlugin` and `fwk.PreEnqueuePlugin`:
- **fh**: Framework handle
- **fts**: Feature flags
- **args**: Configuration arguments (MinCandidateNodesPercentage, MinCandidateNodesAbsolute)
- **Evaluator**: Preemption evaluator for finding victims
- **IsEligiblePod**: Customizable function for filtering eligible victims
- **MoreImportantPod**: Customizable function for sorting pods by importance

### Customization Hooks
- **IsEligiblePodFunc**: Determines if a pod can be preempted (beyond priority check)
- **MoreImportantPodFunc**: Defines pod importance ordering for victim selection

## Key Functions

- **New(ctx, args, handle, features)**: Creates a new preemption plugin
- **PostFilter(ctx, state, pod, nodeStatusMap)**: Main preemption entry point
- **PreEnqueue(ctx, pod)**: Blocks pods with ongoing async preemption
- **GetOffsetAndNumCandidates(numNodes)**: Calculates preemption search parameters
- **CandidatesToVictimsMap(candidates)**: Maps candidate nodes to their victims

## Preemption Flow
1. PostFilter is called when a pod can't be scheduled
2. Evaluator.Preempt finds candidate nodes where preemption could help
3. For each candidate, identifies victim pods to evict
4. Selects best candidate based on number of victims, priority, etc.
5. Nominates the preemptor to the chosen node
6. Returns result for the scheduler to handle victim eviction

## Design Pattern
- Supports async preemption via feature flag
- Randomized offset for fair node selection
- Configurable minimum candidate nodes (percentage or absolute)
- Extensible via IsEligiblePod and MoreImportantPod hooks
