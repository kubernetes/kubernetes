# Package: pod

## Purpose
Core utilities for Pod API objects including container iteration, secret/configmap discovery, condition management, feature-gated field handling, validation options, and deprecation warnings.

## Key Types
- `ContainerType` - Bitmask enum: `Containers`, `InitContainers`, `EphemeralContainers`, `AllContainers`
- `ContainerVisitor` - Function type for visiting containers
- `Visitor` - Function type for visiting named resources

## Key Functions

### Container Iteration
- `VisitContainers(podSpec, mask, visitor)` - Iterates over containers matching the mask
- `ContainerIter(podSpec, mask)` - Returns Go 1.23+ iterator over containers
- `VisitPodSecretNames(pod, visitor, containerType)` - Finds all secret references in a pod
- `VisitPodConfigmapNames(pod, visitor, containerType)` - Finds all configmap references in a pod

### Pod Status Helpers
- `IsPodReady(pod)` / `IsPodReadyConditionTrue(status)` - Check pod readiness
- `GetPodReadyCondition(status)` / `GetPodCondition(status, type)` - Extract conditions
- `UpdatePodCondition(status, condition)` - Update or add a condition

### Feature Gate Handling
- `DropDisabledPodFields(pod, oldPod)` - Removes disabled feature fields from pods
- `DropDisabledTemplateFields(podTemplate, oldPodTemplate)` - Same for PodTemplateSpec
- `GetValidationOptionsFromPodSpecAndMeta(...)` - Returns validation options based on features and existing values

### Warnings
- `GetWarningsForPod(ctx, pod, oldPod)` - Generates deprecation warnings
- `GetWarningsForPodTemplate(ctx, fieldPath, template, oldTemplate)` - Same for templates

### Other Utilities
- `IsRestartableInitContainer(container)` - Checks for sidecar container pattern
- `HasAPIObjectReference(pod)` - Checks if pod references other API objects
- `ApparmorFieldForAnnotation(annotation)` - Converts legacy AppArmor annotation to field

## Feature Gates Handled
UserNamespacesSupport, SupplementalGroupsPolicy, PodLevelResources, InPlacePodVerticalScaling, SidecarContainers, DynamicResourceAllocation, RecursiveReadOnlyMounts, SELinuxChangePolicy, ImageVolume, and many more.
