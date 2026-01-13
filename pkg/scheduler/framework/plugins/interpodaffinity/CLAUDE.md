# Package: interpodaffinity

## Purpose
Implements pod affinity and anti-affinity scheduling constraints. Ensures pods are scheduled near (affinity) or away from (anti-affinity) other pods based on label selectors and topology keys.

## Key Types

### InterPodAffinity
The plugin struct implementing multiple extension points:
- PreFilter, Filter, PreScore, Score, EnqueueExtensions, SignPlugin
- **parallelizer**: For parallel node processing
- **args**: Configuration (HardPodAffinityWeight)
- **sharedLister**: Access to cluster state
- **nsLister**: Namespace lister for cross-namespace affinity

## Extension Points

### PreFilter
- Parses and validates affinity/anti-affinity terms
- Caches parsed terms in CycleState for Filter/Score phases
- Identifies nodes with existing pods that match affinity terms

### Filter
- Checks hard affinity requirements (RequiredDuringSchedulingIgnoredDuringExecution)
- For each node, verifies:
  - Pod affinity: required pods exist in same topology domain
  - Pod anti-affinity: conflicting pods don't exist in same topology domain
  - Existing pod anti-affinity: node's pods don't reject the incoming pod

### PreScore / Score
- Evaluates soft affinity preferences (PreferredDuringSchedulingIgnoredDuringExecution)
- Sums weighted scores across all matching topology domains
- Higher scores for nodes in topologies with more matching pods (affinity)
- Lower scores for nodes in topologies with conflicting pods (anti-affinity)

## Key Concepts

- **Topology Key**: Label key defining topology domains (e.g., hostname, zone)
- **Label Selector**: Identifies which pods the affinity applies to
- **Namespace Selector**: Specifies namespaces for cross-namespace affinity

## Design Pattern
- Pods with inter-pod affinity cannot be signed (affects result caching)
- Uses parallel processing for scoring across nodes
- Queueing hints respond to pod add/delete/label-change events
