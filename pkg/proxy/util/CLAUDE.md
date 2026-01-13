# Package: util

This package provides utility functions shared across kube-proxy backends (iptables, IPVS, nftables).

## Key Types

- `LineBuffer` - Efficient buffer for building iptables/nftables rules line by line
- `LocalTrafficDetector` - Detects if traffic is from local pods vs external sources
- `NodePortAddresses` - Manages which node IPs should be used for NodePort services
- `NetworkInterfacer` - Interface for querying network interfaces

## Key Functions

- `ShouldSkipService()` - Determines if a service should be skipped (e.g., headless)
- `GetLocalEndpointIPs()` - Returns IPs of endpoints local to this node
- `GetClusterIPByFamily()` - Returns the ClusterIP for a specific IP family
- `GetNodeAddresses()` - Returns node addresses filtered by nodePortAddresses config
- `FilterIncorrectIPVersion()` - Filters endpoints by IP family
- `MapIPsByIPFamily()` - Groups IPs by IPv4/IPv6 family

## Design Notes

- Provides common functionality to avoid code duplication across backends
- Handles dual-stack IP family logic consistently
- LineBuffer optimizes rule generation performance
- LocalTrafficDetector supports externalTrafficPolicy: Local
