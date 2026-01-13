# Package: iptables

This package implements the iptables-based kube-proxy backend, which programs iptables rules to implement Kubernetes Service load balancing and network policies.

## Key Types

- `Proxier` - Main iptables proxy implementation, implements proxy.Provider interface
- Manages service port maps, endpoint maps, and iptables rule generation

## Key Functions

- `NewProxier()` - Creates a new iptables-based proxy instance
- `NewDualStackProxier()` - Creates a dual-stack (IPv4+IPv6) proxy using metaproxier
- `SyncProxyRules()` - Main sync loop that regenerates and applies iptables rules
- `CleanupLeftovers()` - Removes stale kube-proxy iptables rules

## Design Notes

- Uses KUBE-SERVICES chain as entry point from PREROUTING and OUTPUT
- Implements session affinity via iptables recent module
- Supports masquerading for cluster-external traffic (SNAT)
- Uses probability-based load balancing across endpoints
- Supports partial syncs for efficiency when only some services change
- Chain naming: KUBE-SVC-xxx for services, KUBE-SEP-xxx for endpoints
- Handles NodePort, LoadBalancer, and ClusterIP service types
