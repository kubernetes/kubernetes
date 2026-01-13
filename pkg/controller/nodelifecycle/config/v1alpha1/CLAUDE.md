# Package: v1alpha1

Provides versioned defaults for NodeLifecycle controller configuration.

## Key Functions

- **RecommendedDefaultNodeLifecycleControllerConfiguration**: Sets recommended defaults:
  - `PodEvictionTimeout`: 5 minutes
  - `NodeMonitorGracePeriod`: 50 seconds (accounts for HTTP/2 health check timeouts)
  - `NodeStartupGracePeriod`: 60 seconds

## Design Patterns

- Uses the Kubernetes defaulting pattern where defaults are applied explicitly rather than in schema conversion.
- Allows consumers to opt-out of defaults by not calling the defaulting function.
- The 50-second NodeMonitorGracePeriod is specifically set to exceed HTTP2_PING_TIMEOUT (30s) + HTTP2_READ_IDLE_TIMEOUT (15s).
