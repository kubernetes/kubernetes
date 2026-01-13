# Package: proxy

## Purpose
The `proxy` package implements kube-proxy, the Kubernetes network proxy that maintains network rules for Service-to-Pod routing.

## Key Interfaces

- **Provider**: Main interface for proxy implementations.
  - `Sync()`: Immediately synchronizes proxy rules.
  - `SyncLoop()`: Runs periodic synchronization.
  - Implements handlers for EndpointSlices, Services, Node topology, and ServiceCIDRs.

## Key Types

- **ServicePortName**: Unique identifier for a service port (namespace/name:port).
- **ServiceEndpoint**: Identifies a service and one of its endpoints.
- **ProxyMode**: Proxy backend type (iptables, ipvs, nftables, kernelspace).

## Subpackages

- **iptables**: Linux iptables-based proxy implementation.
- **ipvs**: Linux IPVS-based proxy implementation.
- **nftables**: Linux nftables-based proxy implementation.
- **config**: Configuration watching and event handling.
- **healthcheck**: Service and proxy health checking.
- **metrics**: Prometheus metrics for proxy operations.
- **conntrack**: Connection tracking management.

## Architecture

1. Watches Services and EndpointSlices from API server.
2. Maintains change trackers for efficient diff calculation.
3. Synchronizes rules to kernel (iptables/ipvs/nftables).
4. Handles topology-aware routing and traffic policies.

## Design Notes

- Multiple proxy modes for different performance/feature tradeoffs.
- Supports dual-stack (IPv4/IPv6) networking.
- Handles session affinity, external traffic policy, internal traffic policy.
- Uses bounded frequency runner to batch rule updates.
