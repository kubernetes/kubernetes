# Package: rangeallocation

## Purpose
Defines the RangeRegistry interface for persistent storage of allocation ranges, used by IP and port allocators.

## Key Types

- **RangeRegistry**: Interface for range allocation storage operations.

## Key Functions (Interface Methods)

- **Get()**: Retrieves current allocation state from storage.
- **CreateOrUpdate(allocation)**: Atomically creates or updates allocation state.

## Design Notes

- Foundation interface for IP allocator (ServiceCIDR) and port allocator (NodePort ranges).
- Provides atomic create-or-update semantics for allocation bitmaps.
- Implementations handle the actual etcd storage operations.
- Used by service/ipallocator and service/portallocator packages.
