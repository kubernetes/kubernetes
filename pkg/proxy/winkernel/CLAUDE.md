# Package: winkernel

This package implements the Windows kernel-mode kube-proxy backend using Windows HNS (Host Networking Service) for service load balancing.

## Key Types

- `Proxier` - Main Windows proxy implementation, implements proxy.Provider interface
- `serviceInfo` - Extended service information including HNS policy IDs
- `endpointInfo` - Extended endpoint information including HNS endpoint IDs
- `hnsNetworkInfo` - Windows HNS network configuration

## Key Functions

- `NewProxier()` - Creates a new Windows HNS-based proxy instance
- `NewDualStackProxier()` - Creates a dual-stack proxy using metaproxier
- `SyncProxyRules()` - Syncs HNS load balancer policies with desired state
- `cleanupStaleLoadbalancers()` - Removes orphaned HNS load balancer policies

## Design Notes

- Uses Windows HCN (Host Compute Network) API v2
- Creates HNS load balancer policies for each service
- Supports DSR (Direct Server Return) for improved performance
- Handles overlay networks for pod-to-pod communication
- Requires Windows Server 2019 or later
- Supports dual-stack (IPv4 + IPv6) via metaproxier
