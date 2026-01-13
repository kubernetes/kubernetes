# Package: events

Defines string constants for Kubernetes event reasons used throughout the kubelet.

## Event Categories

### Container Events
- `CreatedContainer`, `StartedContainer`, `FailedToCreateContainer`, `FailedToStartContainer`
- `KillingContainer`, `PreemptContainer`, `BackOffStartContainer`, `ExceededGracePeriod`

### Pod Events
- `FailedToKillPod`, `FailedToCreatePodContainer`, `FailedToMakePodDataDirectories`
- `NetworkNotReady`, `SandboxChanged`, `FailedCreatePodSandBox`
- Resize events: `ResizeDeferred`, `ResizeInfeasible`, `ResizeCompleted`, `ResizeStarted`, `ResizeError`

### Image Events
- `PullingImage`, `PulledImage`, `FailedToPullImage`, `FailedToInspectImage`
- `ErrImageNeverPullPolicy`, `BackOffPullImage`

### Node/Kubelet Events
- `NodeReady`, `NodeNotReady`, `NodeSchedulable`, `NodeNotSchedulable`
- `StartingKubelet`, `NodeRebooted`, `NodeShutdown`
- Volume events: `FailedAttachVolume`, `FailedMountVolume`, `VolumeResizeFailed`, etc.

### Probe Events
- `ContainerUnhealthy`, `ContainerProbeWarning`

### Lifecycle Hook Events
- `FailedPostStartHook`, `FailedPreStopHook`

## Key Functions

- `PodResizeCompletedMsg()`: Generates resize completed event message with resource summary
- `PodResizeStartedMsg()`: Generates resize started event message
- `PodResizeErrorMsg()`: Generates resize error event message with error details
- `PodResizePendingMsg()`: Generates resize pending event message
