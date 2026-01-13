# Package: preemption

Handles preemption of pods to admit critical pods when resources are insufficient.

## Key Types

- **CriticalPodAdmissionHandler**: Implements AdmissionFailureHandler to evict pods when critical pods fail admission due to insufficient resources.
- **admissionRequirement**: Tracks resource name and quantity needed.
- **admissionRequirementList**: List of requirements with distance calculation.

## Key Functions

- `NewCriticalPodAdmissionHandler()`: Creates handler with pod getter and killer functions.
- `HandleAdmissionFailure()`: Main entry point - if admission fails only due to resources, evicts pods to make room for critical pod.
- `evictPodsToFreeRequests()`: Evicts pods to free required resources.
- `getPodsToPreempt()`: Finds minimal set of pods to evict.
- `getPodsToPreemptByDistance()`: Chooses pods minimizing "distance" to requirements.
- `sortPodsByQOS()`: Separates pods by QoS class (BestEffort, Burstable, Guaranteed).
- `smallerResourceRequest()`: Compares pod resource requests.

## Eviction Algorithm

1. Only triggers for critical pods with resource-only failures
2. Sorts pods by QoS class (evict BestEffort before Burstable before Guaranteed)
3. Within each class, minimizes number of pods evicted
4. Uses "distance" metric based on fraction of requirements satisfied
5. Prefers evicting pods with smaller resource requests when equidistant

## Design Notes

- Only handles InsufficientResourceError failures; other failures returned unchanged
- Evicted pods get PodFailed status with PreemptContainer reason
- Adds DisruptionTarget condition with PodReasonTerminationByKubelet
- Records PreemptContainer warning events
- Tracks preemption metrics by resource name
