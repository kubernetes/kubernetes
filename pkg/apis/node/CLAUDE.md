# Package: node

## Purpose
Defines the internal (unversioned) API types for the node.k8s.io API group, primarily the RuntimeClass resource for container runtime configuration.

## Key Types

### RuntimeClass
Defines a class of container runtime supported in the cluster. Referenced in PodSpec to determine which container runtime runs all containers in a pod.
- `Handler`: The underlying runtime handler name (e.g., "runc"), must be a DNS label, immutable
- `Overhead`: Optional resource overhead for pods using this runtime
- `Scheduling`: Optional scheduling constraints for nodes supporting this runtime

### Overhead
Resource overhead associated with running a pod:
- `PodFixed`: Fixed resource overhead (CPU, memory) added to pod resource calculations

### Scheduling
Scheduling constraints for RuntimeClass-aware scheduling:
- `NodeSelector`: Labels that nodes must have to support this RuntimeClass (merged with pod's nodeSelector)
- `Tolerations`: Tolerations appended to pods using this RuntimeClass (duplicates excluded)

### RuntimeClassList
Standard list wrapper for RuntimeClass objects.

## Key Constants/Variables
- `GroupName`: "node.k8s.io"
- `SchemeGroupVersion`: node.k8s.io with internal version

## Key Functions
- `Kind(kind string)`: Returns qualified GroupKind
- `Resource(resource string)`: Returns qualified GroupResource
- `AddToScheme`: Registers types with a scheme

## Notes
- RuntimeClasses are cluster-scoped (not namespaced)
- Kubelet resolves RuntimeClassName before running pods
- See KEP sig-node/585-runtime-class for design details
