# Package checksum

Package checksum provides checksum calculation and verification for checkpoint data integrity.

## Key Types

- `Checksum`: uint64 type representing a FNV-1a hash of checkpoint data

## Key Functions

- `New(data interface{}) Checksum`: Calculates checksum of any data structure using deep hash
- `Verify(data interface{}) error`: Compares stored checksum against recalculated checksum

## Implementation

- Uses FNV-1a 32-bit hash algorithm (fast, good distribution)
- Leverages `hashutil.DeepHashObject` for consistent hashing of complex structures
- Returns `CorruptCheckpointError` on verification failure

## Design Notes

- Lightweight integrity check for detecting checkpoint corruption
- Works with any Go data structure via reflection-based deep hashing
- Checksum stored as uint64 for efficient storage and comparison
