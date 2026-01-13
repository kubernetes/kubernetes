# Package: kubemark

## Purpose
The `kubemark` package provides a lightweight ("hollow") kubelet and controller for simulating large Kubernetes clusters for scalability testing.

## Key Types

### HollowKubelet
- Lightweight kubelet implementation using fake/mock dependencies.
- Uses fake OS interface, fake mounter, fake container runtime.
- Runs a real kubelet with minimal resource overhead.

### KubemarkController
- Manages hollow nodes in a kubemark cluster.
- Creates/deletes nodes by managing ReplicationControllers.
- Supports node groups with labels for autoscaler integration.

## Key Functions

- **NewHollowKubelet**: Creates a hollow kubelet with fake dependencies.
- **GetHollowKubeletConfig**: Builds KubeletConfiguration with appropriate defaults.
- **NewKubemarkController**: Creates controller for managing hollow nodes.
- **SetNodeGroupSize**: Scales node groups up or down.
- **GetNodeGroupSize/GetNodeGroupTargetSize**: Returns current/target node count.

## Architecture

1. **External Cluster**: Hosts kubemark pods (hollow nodes run as pods).
2. **Kubemark Cluster**: The simulated cluster with hollow nodes.
3. **Node Groups**: Nodes organized by labels for autoscaling.

## Design Notes

- Used for Kubernetes scalability testing (10k+ node simulation).
- Hollow kubelets don't actually run containers.
- Integrates with cluster autoscaler via node group APIs.
- Requires at least one existing hollow node as template.
- Uses ReplicationControllers to manage hollow node pods.
