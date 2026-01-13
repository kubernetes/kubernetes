# Package: util

Utility functions for the kuberuntime package, primarily for pod sandbox and namespace handling.

## Key Functions

- **PodSandboxChanged(pod, podStatus)**: Checks if a pod's sandbox needs to be recreated. Returns (changed, attempt, sandboxID). Triggers recreation when:
  - No sandbox exists
  - Multiple ready sandboxes exist
  - Latest sandbox is not ready
  - Network namespace changed
  - Non-host-network pod sandbox has no IP

- **IpcNamespaceForPod(pod)**: Returns the IPC namespace mode (NODE or POD) based on pod.Spec.HostIPC.

- **NetworkNamespaceForPod(pod)**: Returns the network namespace mode (NODE or POD) based on pod.Spec.HostNetwork.

- **PidNamespaceForPod(pod)**: Returns the PID namespace mode:
  - NODE if HostPID is true
  - POD if ShareProcessNamespace is true
  - CONTAINER otherwise (default)

- **NamespacesForPod(pod, runtimeHelper, rcManager)**: Builds complete NamespaceOption including IPC, Network, PID, and user namespace mappings.

## Key Types

- **RuntimeHandlerResolver**: Interface for looking up runtime handlers from runtime class names.

## Design Notes

- Namespace modes map to CRI runtimeapi.NamespaceMode values
- User namespace support requires RuntimeClass handler lookup
