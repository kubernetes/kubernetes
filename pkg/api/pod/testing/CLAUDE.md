# Package: testing

## Purpose
Test helper package for constructing Pod and Container API objects that pass validation, providing a fluent builder pattern for unit tests.

## Key Types
- `Tweak` - Function type `func(*api.Pod)` for modifying Pod objects
- `TweakContainer` - Function type `func(*api.Container)` for modifying Container objects
- `TweakPodStatus` - Function type `func(*api.PodStatus)` for modifying PodStatus objects

## Key Functions

### Pod Construction
- `MakePod(name string, tweaks ...Tweak) *api.Pod` - Creates a valid Pod with defaults (single container, DNSClusterFirst, RestartPolicyAlways)
- `MakePodSpec(tweaks ...Tweak) api.PodSpec` - Creates just the PodSpec portion

### Pod Tweaks (Setters)
- `SetNamespace`, `SetResourceVersion`, `SetNodeName`, `SetNodeSelector`
- `SetContainers`, `SetInitContainers`, `SetEphemeralContainers`, `SetVolumes`
- `SetSecurityContext`, `SetAffinity`, `SetTolerations`, `SetDNSPolicy`, `SetDNSConfig`
- `SetAnnotations`, `SetLabels`, `SetStatus`, `SetResourceClaims`
- And many more for comprehensive pod configuration

### Container Construction
- `MakeContainer(name string, tweaks ...TweakContainer) api.Container` - Creates a valid container with defaults
- `MakeResourceRequirements(requests, limits map[string]string) api.ResourceRequirements` - Builds resource specs

### Container Tweaks
- `SetContainerImage`, `SetContainerResources`, `SetContainerPorts`
- `SetContainerResizePolicy`, `SetContainerSecurityContext`, `SetContainerRestartPolicy`

### Status Construction
- `MakePodStatus(tweaks ...TweakPodStatus) api.PodStatus`
- `MakeContainerStatus(name string, allocatedResources api.ResourceList) api.ContainerStatus`
- `SetContainerStatuses`, `SetInitContainerStatuses`, `SetEphemeralContainerStatuses`

## Design Pattern
Functional options pattern allows composable, readable test setup while ensuring API-valid defaults.
