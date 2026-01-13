# Package: scheduling

## Purpose
Defines the internal (unversioned) API types for the scheduling.k8s.io API group, including PriorityClass for pod scheduling priority and Workload for advanced scheduling policies.

## Key Types

### PriorityClass
Defines the mapping from a priority class name to a priority value. Cluster-scoped.
- `Value`: Integer priority (higher = more important)
- `GlobalDefault`: If true, this class is used for pods without a priority class
- `Description`: Human-readable description of when to use this class
- `PreemptionPolicy`: PreemptLowerPriority (default) or Never

### Workload (Alpha)
Expresses scheduling constraints for workload lifecycle management including gang scheduling.
- `ControllerRef`: Optional reference to controlling object (Deployment, Job, etc.)
- `PodGroups`: List of pod groups with scheduling policies (max 8)

### PodGroup
Represents a set of pods with a common scheduling policy.
- `Name`: Unique identifier within the Workload (DNS label)
- `Policy`: One of Basic or Gang scheduling

### PodGroupPolicy
- `Basic`: Standard Kubernetes scheduling behavior
- `Gang`: All-or-nothing scheduling with MinCount

## Key Constants
- `DefaultPriorityWhenNoDefaultClassExists`: 0
- `HighestUserDefinablePriority`: 1,000,000,000 (1 billion)
- `SystemCriticalPriority`: 2,000,000,000 (2 billion)
- `SystemPriorityClassPrefix`: "system-"

## System Priority Classes
- `system-node-critical`: Value 2,000,001,000 - Must not be moved from current node
- `system-cluster-critical`: Value 2,000,000,000 - Critical but can be moved if necessary

## Key Functions
- `Kind(kind string)`: Returns qualified GroupKind
- `Resource(resource string)`: Returns qualified GroupResource

## Notes
- Priority values > 1 billion reserved for Kubernetes system use
- GangSchedulingPolicy enables all-or-nothing scheduling for batch workloads
