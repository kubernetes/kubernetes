# Package: config

Contains configuration types for the Pod GC controller.

## Key Types

- **PodGCControllerConfiguration**: Configuration struct containing:
  - `TerminatedPodGCThreshold`: Maximum number of terminated pods allowed before GC kicks in.

## Design Patterns

- Simple configuration with single threshold parameter.
- Part of the componentconfig pattern for kube-controller-manager.
- When threshold is 0 or negative, terminated pod GC is disabled.
