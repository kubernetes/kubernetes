# Package: nftables

This package implements the nftables-based kube-proxy backend, which uses Linux nftables (the successor to iptables) for service load balancing.

## Key Types

- `Proxier` - Main nftables proxy implementation, implements proxy.Provider interface

## Key Functions

- `NewProxier()` - Creates a new nftables-based proxy instance
- `NewDualStackProxier()` - Creates a dual-stack proxy using metaproxier
- `SyncProxyRules()` - Syncs nftables rules with desired service/endpoint state
- `CleanupLeftovers()` - Removes stale kube-proxy nftables rules

## Design Notes

- nftables provides better performance and more features than iptables
- Uses a single nftables transaction for atomic rule updates
- Supports sets and maps for efficient IP/port matching
- Uses verdict maps for faster packet classification than iptables chains
- Requires Linux kernel 3.13+ and nftables userspace tools
- Provides the same service semantics as iptables mode
- Better suited for clusters with many services due to set-based matching
