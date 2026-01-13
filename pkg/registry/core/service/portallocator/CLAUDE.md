# Package: portallocator

## Purpose
Provides NodePort allocation for Services using a bitmap-based allocator over a port range.

## Key Types

- **Interface**: Port allocator interface with Allocate, AllocateNext, Release, ForEach, Has, Destroy, EnableMetrics.
- **PortAllocator**: Main implementation wrapping allocator.Interface for port management.
- **ErrNotInRange**: Error when requested port is outside valid range.

## Key Functions

- **New(pr, allocatorFactory)**: Creates port allocator for a PortRange.
- **NewInMemory(pr)**: Creates in-memory allocator (for testing/repair).
- **NewFromSnapshot(snap)**: Restores allocator from RangeAllocation snapshot.
- **Allocate(port)**: Reserves specific port number.
- **AllocateNext()**: Allocates next available port using random scan strategy.
- **Release(port)**: Returns port to pool.
- **Snapshot(dst)**: Saves state to RangeAllocation.
- **Restore(pr, data)**: Restores state from snapshot.
- **calculateRangeOffset(pr)**: Computes offset for KEP-3070 range subdivision.

## Design Notes

- Wraps generic allocator.Interface, converting port numbers to offsets.
- KEP-3070: Uses range offset (min 16, max 128) to prefer upper range for dynamic allocation.
- Metrics support for allocation tracking.
- Port range specified as "base-end" (e.g., "30000-32767").
