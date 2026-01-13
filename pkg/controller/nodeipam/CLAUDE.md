# Package: nodeipam

Node IPAM controller for managing node pod CIDR allocations.

## Key Types

- `Controller`: Main controller managing node CIDR assignments
- `ipamController`: Interface for legacy IPAM mode support

## Key Functions

- `NewNodeIpamController()`: Creates the controller with CIDR configuration and allocator type
- `Run()`: Starts the CIDR allocation loop

## Purpose

Assigns pod CIDR ranges to nodes for pod networking. Supports multiple allocation strategies and dual-stack networking.

## Allocator Types

- `RangeAllocatorType`: Internal CIDR range allocator (most common)
- `CloudAllocatorType`: Uses cloud provider for CIDR allocation
- `IPAMFromClusterAllocatorType`: Syncs from cluster to cloud
- `IPAMFromCloudAllocatorType`: Syncs from cloud to cluster

## Key Features

- Dual-stack support with separate IPv4 and IPv6 CIDRs
- Configurable node CIDR mask sizes
- Service CIDR exclusion to avoid conflicts
- Cloud provider integration

## Design Notes

- Validates cluster CIDR vs node CIDR mask size
- Initializes CIDR bitmap from existing node allocations
- Service CIDR ranges are excluded from pod CIDR allocation
