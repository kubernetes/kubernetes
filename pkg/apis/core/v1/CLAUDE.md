# Package: v1

## Purpose
Provides conversion, defaulting, and validation logic for core/v1 API types. Bridges external v1 types from k8s.io/api/core/v1 to internal types.

## Key Functions

### Registration
- **AddToScheme**: Registers v1 types, conversion, and defaulting functions.
- **Resource(resource string)**: Returns Group-qualified GroupResource for v1.

### Defaulting (defaults.go)
- **SetDefaults_ResourceList**: Rounds resource values to milli scale.
- **SetDefaults_ReplicationController**: Copies template labels to selector/labels.
- **SetDefaults_Volume**: Defaults empty VolumeSource to EmptyDir.
- **SetDefaults_Container**: Sets ImagePullPolicy based on tag, TerminationMessagePath/Policy.
- **SetDefaults_Pod**: Sets DNS policy, restart policy, service account, security context.
- **SetDefaults_Service**: Sets session affinity, type, IP families, ports.
- **SetDefaults_PersistentVolume/Claim**: Sets reclaim policy, volume mode.
- **SetDefaults_Probe**: Sets timeout, period, thresholds.
- **SetDefaults_Namespace**: Adds metadata.name label.

### Conversion (conversion.go)
- **AddFieldLabelConversionsFor***: Registers field label selectors for Pod, Node, Event, Namespace, Secret, Service.
- Field selector support for common query patterns (metadata.name, spec.nodeName, status.phase, etc.).

## Design Notes

- External types are in k8s.io/api/core/v1, not this package.
- Generated conversion in zz_generated.conversion.go.
- Generated defaults in zz_generated.defaults.go.
- Generated validation in zz_generated.validations.go.
