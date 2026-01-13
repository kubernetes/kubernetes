# Package: ipam

CIDR allocation implementation for node pod networks.

## Key Types

- `CIDRAllocator`: Interface for CIDR allocation/release operations
- `CIDRAllocatorType`: Enum for allocator types (RangeAllocator, CloudAllocator, etc.)
- `CIDRAllocatorParams`: Parameters for creating allocators
- `rangeAllocator`: Implementation using internal CIDR sets

## Key Functions

- `New()`: Factory function creating the appropriate allocator type
- `NewCIDRRangeAllocator()`: Creates a range-based allocator
- `AllocateOrOccupyCIDR()`: Assigns CIDR to a node or marks existing as used
- `ReleaseCIDR()`: Releases CIDR when node is deleted

## Purpose

Provides the CIDR allocation logic for assigning pod network ranges to nodes. The range allocator maintains a bitmap of allocated CIDRs and assigns the next available range to new nodes.

## Key Features

- Supports multiple cluster CIDRs for dual-stack
- Excludes service CIDR ranges from allocation
- Initializes from existing node allocations on startup
- Concurrent workers for CIDR updates (30 workers, 3 retries)

## Design Notes

- Uses workqueue for node processing
- cidrSets map 1:1 with clusterCIDRs for tracking allocation
- Filters out service ranges to prevent IP conflicts
