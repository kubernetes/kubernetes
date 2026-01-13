# Package: node

## Purpose
Provides utilities for working with Kubernetes Node objects, particularly around addresses and readiness.

## Key Types
- `NoMatchError` - Error returned when no preferred address type is found

## Key Functions
- `GetPreferredNodeAddress()` - Returns node address based on preference order
- `GetNodeHostIPs()` - Returns node's primary IP(s) for single or dual-stack clusters
- `IsNodeReady()` - Checks if node has Ready condition set to True

## Constants
- `NodeUnreachablePodReason` - "NodeLost" reason for pods on unreachable nodes
- `NodeUnreachablePodMessage` - Message template for unreachable node pods

## Design Patterns
- Address preference allows callers to specify InternalIP vs ExternalIP order
- Dual-stack support returns up to 2 IPs (one per family)
- Internal IPs are preferred over external for host IPs
- Used by kubelet and controllers for node communication
