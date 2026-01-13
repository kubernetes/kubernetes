# Package: allocator

## Purpose
Provides a generic bitmap-based allocator for managing allocation of contiguous resources like IP addresses and ports.

## Key Types

- **Interface**: Core allocator interface with Allocate, AllocateNext, Release, ForEach, Has, Free, Destroy methods.
- **Snapshottable**: Extended interface adding Snapshot/Restore for persistence.
- **AllocationBitmap**: Bitmap implementation using big.Int for storage.
- **bitAllocator**: Internal strategy interface for choosing next available item.
- **randomScanStrategy**: Picks random offset then scans forward for free item.
- **randomScanStrategyWithOffset**: Subdivides range into upper (preferred) and lower subranges.

## Key Functions

- **NewAllocationMap(max, rangeSpec)**: Creates allocator with random scan strategy.
- **NewAllocationMapWithOffset(max, rangeSpec, offset)**: Creates allocator preferring upper subrange for dynamic allocation.
- **Allocate(offset)**: Reserves specific item by offset.
- **AllocateNext()**: Reserves next available item using strategy.
- **Release(offset)**: Returns item to pool.
- **Snapshot()**: Returns rangeSpec and bitmap bytes for persistence.
- **Restore(rangeSpec, data)**: Restores allocator state from snapshot.

## Design Notes

- Thread-safe using mutex.
- KEP-3070: Range offset reserves lower addresses for static allocation.
- Uses big.Int for efficient bitmap operations.
- ForEach iterates over set bits efficiently.
