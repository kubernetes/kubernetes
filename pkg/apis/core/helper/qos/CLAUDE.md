# Package: qos

## Purpose
Provides Quality of Service (QoS) class computation for pods using internal core API types. Determines whether a pod is Guaranteed, Burstable, or BestEffort.

## Key Functions

- **GetPodQOS(pod)**: Returns the QoS class from pod.Status.QOSClass if set, otherwise computes it via ComputePodQOS.
- **ComputePodQOS(pod)**: Evaluates all containers to determine QoS class based on resource requests/limits.

## QoS Classes

- **BestEffort**: No containers have CPU or memory requests/limits.
- **Guaranteed**: All containers have CPU and memory limits set, and requests equal limits.
- **Burstable**: Everything else (some resources specified but not meeting Guaranteed criteria).

## Algorithm

1. If PodLevelResources feature is enabled and pod.Spec.Resources is set, use pod-level resources.
2. Otherwise, aggregate requests/limits from all containers (init + regular, excluding ephemeral).
3. If no requests and no limits: BestEffort.
4. If all containers have both CPU and memory limits, and requests == limits: Guaranteed.
5. Otherwise: Burstable.

## Design Notes

- Only considers CPU and memory (supportedQoSComputeResources).
- Ephemeral containers are excluded from QoS calculation.
- GetPodQOS is preferred over ComputePodQOS for performance when status is populated.
- Supports PodLevelResources feature gate for pod-level resource specifications.
