# Package: qos

Computes OOM score adjustments for containers based on QoS class and resource requests.

## Constants

- `KubeletOOMScoreAdj`: -999 (kubelet process)
- `KubeProxyOOMScoreAdj`: -999 (kube-proxy process)
- `guaranteedOOMScoreAdj`: -997 (Guaranteed pods)
- `besteffortOOMScoreAdj`: 1000 (BestEffort pods)

## Key Functions

- `GetContainerOOMScoreAdjust(pod, container, memoryCapacity)`: Calculates OOM score adjustment for a container based on:
  - Pod QoS class (Guaranteed, Burstable, BestEffort)
  - Container memory request relative to node capacity
  - Whether container is a sidecar
  - Pod-level resource requests (when PodLevelResources feature enabled)

## OOM Score Calculation

- **Guaranteed / Critical**: -997 (last to be killed)
- **BestEffort**: 1000 (first to be killed)
- **Burstable**: Scales between based on memory request percentage
  - Formula: `1000 - (1000 * memRequest / memoryCapacity)`
  - Higher request = lower OOM score = less likely to be killed
  - Minimum score is 3 (guaranteedOOMScoreAdj + 1000 + 1)

## Sidecar Handling

Sidecar containers get OOM score equal to or lower than the minimum regular container score, ensuring sidecars are not killed before the containers they support.

## Design Notes

- Higher OOM score = more likely to be killed when memory is exhausted
- Score range is -1000 to 1000; kernel adds process memory percentage
- Supports PodLevelResources feature for pod-level memory requests
