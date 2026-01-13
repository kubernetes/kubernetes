# Package: config

## Purpose
The `config` package defines the configuration types for kube-proxy.

## Key Types

### KubeProxyConfiguration
Main configuration structure containing:
- **Linux/Windows**: Platform-specific settings.
- **Mode**: Proxy mode (iptables, ipvs, nftables, kernelspace).
- **IPTables/IPVS/NFTables/Winkernel**: Backend-specific configuration.
- **DetectLocalMode**: How to detect local traffic.
- **SyncPeriod/MinSyncPeriod**: Rule sync intervals.
- **NodePortAddresses**: CIDRs for NodePort binding.
- **HealthzBindAddress/MetricsBindAddress**: Server endpoints.

### ProxyMode
- `iptables`: Linux iptables backend.
- `ipvs`: Linux IPVS backend.
- `nftables`: Linux nftables backend.
- `kernelspace`: Windows backend.

### LocalMode
- `ClusterCIDR`: Detect by cluster CIDR.
- `NodeCIDR`: Detect by node CIDR.
- `BridgeInterface`: Detect by bridge interface.
- `InterfaceNamePrefix`: Detect by interface name prefix.

### Supporting Types
- **KubeProxyConntrackConfiguration**: Connection tracking settings.
- **KubeProxyIPTablesConfiguration**: iptables-specific settings.
- **KubeProxyIPVSConfiguration**: IPVS-specific settings (scheduler, timeouts).
- **KubeProxyWinkernelConfiguration**: Windows HNS settings.
- **DetectLocalConfiguration**: Local traffic detection settings.

## Design Notes

- Configuration loaded from file or command-line flags.
- Supports component-base logging and client connection config.
- Platform-specific sections for Linux vs Windows.
