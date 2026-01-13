# Package: plugins

## Purpose
Provides the in-tree plugin registry for the Kubernetes scheduler. This package registers all built-in scheduling plugins and creates the registry used by the scheduler framework.

## Key Functions

### NewInTreeRegistry()
Returns a `runtime.Registry` containing all built-in scheduler plugins:

**Resource Plugins:**
- `DynamicResources`: Handles ResourceClaim allocation for DRA
- `NodeResourcesFit`: Checks if node has sufficient resources
- `NodeResourcesBalancedAllocation`: Scores nodes for balanced resource usage

**Affinity/Anti-Affinity Plugins:**
- `NodeAffinity`: Matches pods to nodes based on node labels
- `InterPodAffinity`: Handles pod affinity/anti-affinity rules
- `PodTopologySpread`: Enforces topology spread constraints

**Node Selection Plugins:**
- `NodeName`: Filters by pod.spec.nodeName
- `NodeUnschedulable`: Filters nodes marked unschedulable
- `NodePorts`: Checks for port conflicts
- `TaintToleration`: Matches pod tolerations to node taints
- `NodeDeclaredFeatures`: Checks node feature requirements

**Volume Plugins:**
- `VolumeBinding`: Handles PV/PVC binding
- `VolumeZone`: Ensures volume zone compatibility
- `VolumeRestrictions`: Checks volume access mode conflicts
- `NodeVolumeLimits` (CSI): Enforces per-node volume limits

**Scheduling Control Plugins:**
- `DefaultBinder`: Binds pods to nodes
- `DefaultPreemption`: Handles pod preemption
- `PrioritySort` (QueueSort): Sorts pods by priority
- `SchedulingGates`: Blocks pods with scheduling gates
- `GangScheduling`: Enables all-or-nothing pod group scheduling
- `ImageLocality`: Scores nodes with cached container images

## Design Pattern
- Uses `runtime.FactoryAdapter` to inject feature flags into plugin constructors
- Plugins can be disabled/replaced via scheduler configuration
- Out-of-tree plugins can be added via `WithFrameworkOutOfTreeRegistry` option
