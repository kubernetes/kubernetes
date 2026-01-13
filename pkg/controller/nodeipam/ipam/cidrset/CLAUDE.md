# Package: cidrset

CIDR set data structure for managing IP range allocation.

## Key Types

- `CidrSet`: Manages a set of CIDR ranges using a bitmap for allocation tracking

## Key Constants

- `clusterSubnetMaxDiff`: 16 - maximum difference between cluster and node mask sizes
- `ErrCIDRRangeNoCIDRsRemaining`: Error when no CIDRs available
- `ErrCIDRSetSubNetTooBig`: Error when subnet mask is too large

## Key Functions

- `NewCIDRSet()`: Creates a new CidrSet from cluster CIDR and node mask size
- `AllocateNext()`: Allocates the next free CIDR range
- `Occupy()`: Marks a specific CIDR as allocated
- `Release()`: Releases a CIDR back to the pool

## Purpose

Provides efficient bitmap-based tracking of CIDR range allocations. Used by the range allocator to manage which pod CIDRs have been assigned to nodes.

## Key Features

- Supports both IPv4 and IPv6
- Thread-safe with mutex protection
- Tracks allocation via big.Int bitmap
- Handles next-candidate optimization for faster allocation

## Design Notes

- IPv6 subnet mask must be >= 48 due to bitmap size limits
- Uses binary math for efficient CIDR-to-index conversion
- Registers Prometheus metrics for allocation tracking
