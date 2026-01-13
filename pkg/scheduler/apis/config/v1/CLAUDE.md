# Package: v1

## Purpose
Provides conversion, defaulting, and registration for the v1 versioned scheduler configuration API. Handles translation between the external API (v1) and internal types.

## Key Functions

### Default Plugins
- **getDefaultPlugins()**: Returns the default plugin configuration for all extension points
- **applyFeatureGates(plugins)**: Adds feature-gated plugins (DynamicResources, GangScheduling, NodeDeclaredFeatures)
- **mergePlugins(default, custom)**: Merges user-provided plugins with defaults

### Default Plugin Set (MultiPoint)
- SchedulingGates, PrioritySort
- NodeUnschedulable, NodeName, TaintToleration, NodeAffinity
- NodePorts, NodeResourcesFit, VolumeRestrictions
- NodeVolumeLimits, VolumeBinding, VolumeZone
- PodTopologySpread, InterPodAffinity
- DefaultPreemption, NodeResourcesBalancedAllocation
- ImageLocality, DefaultBinder

### Defaulting
- **SetDefaults_KubeSchedulerConfiguration**: Sets defaults for the full config
- **setDefaults_KubeSchedulerProfile**: Sets defaults for each profile
- Default resources: CPU and Memory with weight 1

### Plugin Config Defaults
- Automatically creates default PluginConfig for enabled plugins
- Applies scheme defaults to plugin arguments

## Key Types

- **pluginIndex**: Tracks plugin position during merging
- **defaultResourceSpec**: Default resources for scoring (CPU, Memory)

## Design Pattern
- Uses k8s.io/apimachinery scheme for type registration
- Generated code (zz_generated.*.go) handles boilerplate
- Feature gates control which plugins are included by default
