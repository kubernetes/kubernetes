# Package: nodeaffinity

## Purpose
Implements node affinity and node selector scheduling constraints. Matches pods to nodes based on node labels, supporting both required and preferred scheduling terms.

## Key Types

### NodeAffinity
The plugin struct implementing multiple extension points:
- PreFilter, Filter, PreScore, Score, EnqueueExtensions, SignPlugin
- **handle**: Framework handle
- **addedNodeSelector**: Scheduler-enforced node selector (from config)
- **addedPrefSchedTerms**: Scheduler-enforced preferred terms

## Extension Points

### PreFilter
- Parses pod's nodeSelector and nodeAffinity
- Merges with scheduler-enforced selectors
- Detects conflicting affinity terms (returns UnschedulableAndUnresolvable)
- Caches RequiredNodeAffinity in CycleState

### Filter
- Evaluates required node affinity (RequiredDuringSchedulingIgnoredDuringExecution)
- Checks pod.spec.nodeSelector matches node labels
- Checks scheduler-enforced node selector
- Returns UnschedulableAndUnresolvable if no match

### PreScore
- Parses preferred scheduling terms
- Combines pod preferences with scheduler-enforced preferences
- Caches in CycleState for Score phase

### Score
- Evaluates preferred node affinity (PreferredDuringSchedulingIgnoredDuringExecution)
- Sums weights of matching preference terms
- Returns weighted score for node selection

## Key Functions

- **SignPod(ctx, pod)**: Returns node affinity and node selector for pod signing
- **isSchedulableAfterNodeChange**: Queueing hint for node label changes

## Configuration
- Supports scheduler-level addedAffinity in KubeSchedulerConfiguration
- Scheduler-enforced selectors are AND'd with pod selectors
- Scheduler-enforced preferences are added to pod preferences

## Design Pattern
- Separates required (Filter) from preferred (Score) matching
- Supports both legacy nodeSelector and nodeAffinity API
- Responds to Node/Add and Node/UpdateNodeLabel events
