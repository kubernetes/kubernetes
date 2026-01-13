# Package: storage

## Purpose
Provides etcd-backed storage for allocation bitmaps, wrapping an in-memory allocator with persistent state.

## Key Types

- **Etcd**: Persistent allocator that wraps allocator.Snapshottable and stores state in etcd via RangeAllocation objects.

## Key Functions

- **NewEtcd(alloc, baseKey, config)**: Creates etcd-backed allocator using raw storage interface.
- **Allocate(offset)**: Allocates item and persists state atomically.
- **AllocateNext()**: Allocates next available item and persists state.
- **Release(item)**: Releases item and persists updated state.
- **Get()**: Returns current RangeAllocation from storage.
- **CreateOrUpdate(snapshot)**: Atomically creates or updates allocation state.
- **tryUpdate(fn)**: Performs read-update cycle with GuaranteedUpdate for atomic operations.

## Design Notes

- Implements both allocator.Interface and rangeallocation.RangeRegistry.
- Uses optimistic concurrency with ResourceVersion tracking.
- Restores in-memory state from etcd on version mismatch.
- Atomic updates via storage.SimpleUpdate pattern.
- Used by both IP and port allocators for persistent storage.
