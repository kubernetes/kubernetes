# Package bitmask

Package bitmask provides a 64-bit bitmask implementation for representing NUMA node affinity in topology hints.

## Key Types

- `BitMask`: Interface for NUMA affinity bitmask operations
- `bitMask`: uint64-based implementation

## BitMask Interface Methods

Modification:
- `Add(bits...)`: Set specified bits to 1
- `Remove(bits...)`: Clear specified bits
- `And(masks...)`: Bitwise AND with other masks
- `Or(masks...)`: Bitwise OR with other masks
- `Clear()`: Reset all bits to 0
- `Fill()`: Set all bits to 1

Query:
- `IsSet(bit)`: Check if specific bit is set
- `AnySet(bits)`: Check if any of the bits are set
- `IsEmpty()`: Check if all bits are zero
- `IsEqual(mask)`: Compare equality
- `IsNarrowerThan(mask)`: Compare narrowness (fewer bits set)
- `IsLessThan/IsGreaterThan(mask)`: Numeric comparison
- `Count()`: Number of bits set
- `GetBits()`: List of set bit positions

## Package Functions

- `NewEmptyBitMask()`: Create empty mask
- `NewBitMask(bits...)`: Create mask with specified bits
- `And(first, masks...)`: Package-level AND operation
- `Or(first, masks...)`: Package-level OR operation
- `IterateBitMasks(bits, callback)`: Iterate all possible submasks

## Design Notes

- Supports up to 64 NUMA nodes (bit positions 0-63)
- Narrower masks preferred (fewer NUMA nodes = better locality)
- Used to represent and merge topology hints across providers
