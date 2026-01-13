# Package: names

## Purpose
Defines string constants for all built-in scheduler plugin names. Centralizes plugin naming to ensure consistency across the codebase.

## Plugin Name Constants

### Queue Management
- **PrioritySort**: Queue sorting by pod priority

### Binding
- **DefaultBinder**: Standard pod-to-node binding

### Preemption
- **DefaultPreemption**: Standard preemption logic

### Resource Management
- **DynamicResources**: Dynamic Resource Allocation (DRA)
- **NodeResourcesFit**: Node resource capacity checking
- **NodeResourcesBalancedAllocation**: Balanced resource scoring

### Affinity/Topology
- **NodeAffinity**: Node affinity/selector matching
- **InterPodAffinity**: Pod affinity/anti-affinity
- **PodTopologySpread**: Topology spread constraints

### Node Selection
- **NodeName**: Direct node name matching
- **NodeUnschedulable**: Unschedulable node filtering
- **NodePorts**: Host port conflict checking
- **TaintToleration**: Taint/toleration matching
- **NodeDeclaredFeatures**: Node feature requirements

### Volume Management
- **VolumeBinding**: PV/PVC binding
- **VolumeZone**: Volume topology zone matching
- **VolumeRestrictions**: Volume access mode checking
- **NodeVolumeLimits**: Per-node volume count limits

### Other
- **ImageLocality**: Image caching preference
- **SchedulingGates**: Scheduling gate enforcement
- **GangScheduling**: Pod group scheduling

## Usage
These constants are used:
- In plugin registry construction
- In scheduler configuration files
- In plugin implementations (Name() method)
- In logs and metrics
