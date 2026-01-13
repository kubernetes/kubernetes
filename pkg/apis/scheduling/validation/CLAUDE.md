# Package: scheduling/validation

## Purpose
Provides validation logic for scheduling API types, including PriorityClass and Workload resources.

## Key Functions
- `ValidatePriorityClass(pc *scheduling.PriorityClass)`: Validates PriorityClass fields including name, value, preemption policy, and system priority class rules
- `ValidatePriorityClassUpdate(pc, oldPc *scheduling.PriorityClass)`: Validates updates, ensuring immutable fields (value, preemptionPolicy) are not changed
- `ValidateWorkload(workload *scheduling.Workload)`: Validates Workload resources including metadata and spec
- `ValidateWorkloadUpdate(workload, oldWorkload *scheduling.Workload)`: Validates Workload updates with immutability checks

## Key Validation Rules
- System priority classes (prefixed with "system-") must be known/predefined
- Non-system priority classes cannot exceed HighestUserDefinablePriority
- Workloads must have at least one PodGroup but no more than WorkloadMaxPodGroups
- PodGroup policies must specify exactly one of: basic or gang scheduling
- GangSchedulingPolicy requires positive minCount

## Design Notes
- Uses field.ErrorList pattern for accumulating validation errors
- Leverages core validation helpers from `k8s.io/kubernetes/pkg/apis/core/validation`
- Enforces Kubernetes naming conventions via apimachineryvalidation
