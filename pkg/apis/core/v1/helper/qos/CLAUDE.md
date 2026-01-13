# Package: qos

## Purpose
Provides Quality of Service (QoS) class computation for pods using external v1 API types (k8s.io/api/core/v1).

## Key Types

- **QOSList**: Map of resource names to QoS classes: `map[v1.ResourceName]v1.PodQOSClass`

## Key Functions

- **GetPodQOS(pod *v1.Pod)**: Returns QoS class from pod.Status.QOSClass if set, otherwise computes it.
- **ComputePodQOS(pod *v1.Pod)**: Evaluates containers to determine QoS class.

## QoS Classes

- **v1.PodQOSBestEffort**: No CPU/memory requests or limits specified.
- **v1.PodQOSGuaranteed**: All containers have CPU and memory limits, requests equal limits.
- **v1.PodQOSBurstable**: Some resources specified but not meeting Guaranteed criteria.

## Algorithm

1. If PodLevelResources feature enabled and pod.Spec.Resources set, use pod-level resources.
2. Otherwise, aggregate from all containers (init + regular containers).
3. Ephemeral containers excluded from QoS calculation.
4. Only CPU and memory considered for QoS determination.

## Design Notes

- Mirrors the internal core/helper/qos package but uses v1 types.
- GetPodQOS is more efficient than ComputePodQOS when status is populated.
- Supports PodLevelResources feature gate for pod-level resource specifications.
