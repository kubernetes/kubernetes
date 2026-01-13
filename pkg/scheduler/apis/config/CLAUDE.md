# Package: config

## Purpose
Defines the internal types for the Kubernetes scheduler configuration API (kubescheduler.config.k8s.io). These types are used internally and converted to/from versioned types.

## Key Types

- **KubeSchedulerConfiguration**: Main configuration struct for the scheduler.
- **KubeSchedulerProfile**: Defines a scheduling profile with name, plugins, and plugin config.
- **Plugins**: Plugin configuration for all extension points.
- **PluginSet**: Enabled/disabled plugins for an extension point.
- **Plugin**: Plugin name and optional weight.
- **PluginConfig**: Arguments for a specific plugin.
- **Extender**: Configuration for external scheduler extenders.
- **ExtenderTLSConfig**: TLS settings for extender communication.
- **ExtenderManagedResource**: Extended resource managed by an extender.

## Extension Points (in Plugins struct)

- **PreEnqueue**: Before adding pods to queue
- **QueueSort**: Sorting pods in queue
- **PreFilter/Filter**: Node filtering
- **PostFilter**: After filtering (e.g., preemption)
- **PreScore/Score**: Node scoring
- **Reserve**: Resource reservation
- **Permit**: Binding approval
- **PreBind/Bind/PostBind**: Pod binding

## Constants

- **DefaultKubeSchedulerPort**: 10259
- **DefaultPercentageOfNodesToScore**: 0 (adaptive)
- **MaxCustomPriorityScore**: 10
- **MaxTotalScore**: math.MaxInt64

## Key Functions

- **Plugins.Names()**: Returns list of all enabled plugin names.

## Design Notes

- Internal types converted from versioned types (v1) during config loading.
- Supports multiple scheduling profiles for different workload types.
- Plugin-based extensibility via framework extension points.
