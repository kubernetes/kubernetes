# Package: podtopologyspread

## Purpose
Implements pod topology spread constraints for distributing pods across topology domains (zones, nodes, etc.). Supports both hard (Filter) and soft (Score) constraints.

## Key Types

### PodTopologySpread
The plugin struct implementing:
- PreFilter, Filter, PreScore, Score, EnqueueExtensions, SignPlugin
- **defaultConstraints**: System or user-defined default constraints
- **systemDefaulted**: Whether system defaults are in use
- **parallelizer**: For parallel node processing

### topologySpreadConstraint
Internal representation of a spread constraint:
- TopologyKey, MaxSkew, MinDomains
- Selector, NodeAffinityPolicy, NodeTaintsPolicy

## System Defaults
When no constraints specified and defaults enabled:
```yaml
- topologyKey: kubernetes.io/hostname
  whenUnsatisfiable: ScheduleAnyway
  maxSkew: 3
- topologyKey: topology.kubernetes.io/zone
  whenUnsatisfiable: ScheduleAnyway
  maxSkew: 5
```

## Extension Points

### PreFilter
- Parses topology spread constraints
- Computes existing pod distribution per topology domain
- Caches state for Filter/Score phases

### Filter
- Enforces DoNotSchedule (hard) constraints
- Checks if placing pod on node would violate maxSkew
- Considers minDomains requirement

### PreScore / Score
- Evaluates ScheduleAnyway (soft) constraints
- Scores based on how well node improves balance
- Prefers nodes that reduce skew

## Key Concepts

- **TopologyKey**: Node label defining topology domains
- **MaxSkew**: Maximum allowed difference in pod count between domains
- **MinDomains**: Minimum number of domains required
- **WhenUnsatisfiable**: DoNotSchedule (hard) or ScheduleAnyway (soft)

## Design Pattern
- Pods with topology constraints cannot be signed (affect global state)
- Uses parallel processing for large clusters
- Supports NodeAffinityPolicy and NodeTaintsPolicy for domain filtering
- MatchLabelKeys feature for per-rollout spreading
