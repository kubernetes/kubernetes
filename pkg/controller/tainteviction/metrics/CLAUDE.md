# Package: metrics

Provides Prometheus metrics for the TaintEviction controller.

## Key Metrics

- **PodDeletionsTotal**: Counter of pods deleted by TaintEvictionController since startup.

- **PodDeletionsLatency**: Histogram tracking time (in seconds) between taint effect activation and pod deletion. Buckets range from 5ms to 4 minutes.

## Design Patterns

- Metrics are ALPHA stability level.
- Latency histogram helps identify eviction delays.
- Registered with the legacy Prometheus registry.
- Uses sync.Once for idempotent registration.
