# Package: config

Node IPAM controller configuration types for the kube-controller-manager.

## Key Types

- `NodeIPAMControllerConfiguration`: Contains configuration elements including:
  - `ServiceCIDR`: Primary service CIDR range
  - `SecondaryServiceCIDR`: Secondary service CIDR for dual-stack
  - `NodeCIDRMaskSize`: Mask size for single-stack clusters
  - `NodeCIDRMaskSizeIPv4`: IPv4 mask size for dual-stack
  - `NodeCIDRMaskSizeIPv6`: IPv6 mask size for dual-stack

## Purpose

Defines the internal configuration structure used by the Node IPAM controller. Configuration determines how pod CIDRs are carved out from cluster CIDRs.

## Design Notes

- NodeCIDRMaskSize is incompatible with dual-stack (use IPv4/IPv6 specific sizes)
- Service CIDR configuration helps avoid IP conflicts
- Part of the component-config pattern used throughout kube-controller-manager
