# Package: pod

## Purpose
Provides utilities for v1.Pod objects including container iteration, resource discovery, status helpers, restart policy evaluation, and availability checks. This is the v1 API version of the pod utilities.

## Key Types
- `ContainerType` - Bitmask enum: `Containers`, `InitContainers`, `EphemeralContainers`, `AllContainers`
- `ContainerVisitor` - Function type for visiting containers
- `Visitor` - Function type for visiting named resources

## Key Functions

### Container Iteration
- `VisitContainers(podSpec, mask, visitor)` - Iterates over containers matching the mask
- `ContainerIter(podSpec, mask)` - Returns Go 1.23+ iterator over containers

### Resource Discovery
- `VisitPodSecretNames(pod, visitor)` - Finds all secret references (ImagePullSecrets, env, volumes)
- `VisitPodConfigmapNames(pod, visitor)` - Finds all configmap references

### Container Status Helpers
- `GetContainerStatus(statuses, name)` - Extracts status by container name
- `GetExistingContainerStatus(statuses, name)` - Same but without found flag
- `GetIndexOfContainerStatus(statuses, name)` - Returns index of container status

### Pod Condition Helpers
- `IsPodReady(pod)` / `IsPodReadyConditionTrue(status)` - Check pod readiness
- `IsContainersReadyConditionTrue(status)` - Check containers ready condition
- `IsPodTerminal(pod)` / `IsPodPhaseTerminal(phase)` - Check if pod is in terminal state
- `IsPodAvailable(pod, minReadySeconds, now)` - Check availability with minReadySeconds
- `GetPodCondition(status, type)` / `GetPodReadyCondition(status)` - Extract conditions
- `UpdatePodCondition(status, condition)` - Update or add a condition

### Restart Policy Helpers
- `IsRestartableInitContainer(container)` - Checks for sidecar container (RestartPolicy=Always)
- `IsContainerRestartable(pod, container)` - Evaluates all restart policies
- `ContainerShouldRestart(container, pod, exitCode)` - Determines if container should restart based on exit code
- `FindMatchingContainerRestartRule(container, exitCode)` - Finds matching restart rule for exit code
- `AllContainersCouldRestart(podSpec)` - Checks if any container has RestartAllContainers action

### Generation Tracking
- `CalculatePodStatusObservedGeneration(pod)` - Calculates observedGeneration for pod status
- `CalculatePodConditionObservedGeneration(status, generation, conditionType)` - Calculates observedGeneration for conditions
