# Package: metrics

Prometheus metrics for the plugin manager.

## Metrics

- **plugin_manager_total_plugins**: Gauge counting plugins in each state
  - Labels: `socket_path`, `state` (actual_state_of_world or desired_state_of_world)
  - Stability: ALPHA

## Key Types

- **totalPluginsCollector**: Custom collector implementing metrics.StableCollector.
- **pluginCount**: Helper map type for counting plugins by state and socket path.

## Key Functions

- `Register(asw, dsw)`: Registers the plugin metrics collector with the legacy registry. Uses sync.Once to ensure single registration.

## Implementation

The collector:
1. Iterates registered plugins from ActualStateOfWorld
2. Iterates plugins to register from DesiredStateOfWorld
3. Reports counts per socket path and state

## Usage

Called by PluginManager.Run() after starting the watcher and reconciler.
