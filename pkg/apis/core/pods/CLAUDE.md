# Package: pods

## Purpose
Helper functions for working with Pod specifications, including container visitor patterns and downward API field label conversion.

## Key Types

- **ContainerVisitorWithPath**: Function type for visiting containers with their field paths.

## Key Functions

- **VisitContainersWithPath(podSpec, specPath, visitor)**: Iterates over all containers (init, regular, ephemeral) in a pod spec, calling the visitor with each container and its field.Path. Returns false if visitor short-circuits.

- **ConvertDownwardAPIFieldLabel(version, label, value)**: Converts downward API field labels from v1 to internal format. Supports:
  - metadata.annotations, metadata.labels, metadata.name, metadata.namespace, metadata.uid
  - spec.nodeName, spec.restartPolicy, spec.serviceAccountName, spec.schedulerName
  - status.phase, status.hostIP, status.hostIPs, status.podIP, status.podIPs
  - Converts deprecated "spec.host" to "spec.nodeName"

## Design Notes

- Container visitor pattern enables consistent iteration over all container types.
- Only supports v1 version for downward API conversion.
- Field labels support subscripted paths for annotations and labels.
