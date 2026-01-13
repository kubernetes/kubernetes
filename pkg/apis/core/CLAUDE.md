# Package: core

## Purpose
Internal (unversioned) API types for the core Kubernetes API group (empty group name ""). This is the in-memory representation of all fundamental Kubernetes resources like Pods, Services, Nodes, ConfigMaps, Secrets, and more.

## Key Types

- **Pod, PodSpec, PodStatus**: Container workload definition with scheduling, networking, storage configuration.
- **Service, ServiceSpec**: Network service abstraction for pod load balancing.
- **Node, NodeSpec, NodeStatus**: Cluster node representation with capacity, conditions, addresses.
- **ConfigMap, Secret**: Configuration and sensitive data storage.
- **PersistentVolume, PersistentVolumeClaim**: Storage provisioning and claims.
- **Namespace**: Logical cluster partitioning.
- **ResourceQuota, LimitRange**: Resource governance.
- **Volume, VolumeSource**: Storage mount abstractions (EmptyDir, HostPath, PVC, CSI, etc.).
- **Container, ContainerPort, ResourceRequirements**: Container specifications.
- **Taint, Toleration**: Node scheduling constraints.

## Key Functions

- **Kind(kind string)**: Returns Group-qualified GroupKind.
- **Resource(resource string)**: Returns Group-qualified GroupResource.
- **AddToScheme**: Registers all core types with a scheme.
- **Taint.MatchTaint()**: Checks if taints match by key:effect.
- **Toleration.MatchToleration()**: Checks toleration matching.
- **ResourceList.CPU(), Memory(), Storage()**: Resource accessor helpers.
- **ObjectReference.SetGroupVersionKind()**: GVK manipulation for references.

## Key Constants

- **NamespaceDefault, NamespaceSystem, NamespacePublic, NamespaceNodeLease**: Well-known namespaces.
- Various annotation keys for image policy, mirrors, tolerations, seccomp, AppArmor.

## Design Notes

- This is the internal version; versioned types (v1) are in k8s.io/api/core/v1.
- Types here are used throughout the Kubernetes codebase for in-memory manipulation.
- Generated deepcopy functions in zz_generated.deepcopy.go.
