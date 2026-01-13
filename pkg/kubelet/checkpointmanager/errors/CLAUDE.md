# Package errors

Package errors defines error types for the checkpoint manager.

## Key Types

- `CorruptCheckpointError`: Indicates checksum mismatch when reading checkpoint data

## Key Variables

- `ErrCheckpointNotFound`: Sentinel error returned when checkpoint key does not exist

## CorruptCheckpointError

Fields:
- `ActualCS`: The checksum calculated from the data
- `ExpectedCS`: The checksum stored in the checkpoint

Methods:
- `Error() string`: Returns "checkpoint is corrupted"
- `Is(target error) bool`: Supports errors.Is() for type matching

## Usage

```go
var csErr *CorruptCheckpointError
if errors.As(err, &csErr) {
    // Handle corruption with access to actual/expected checksums
}
if errors.Is(err, CorruptCheckpointError{}) {
    // Handle corruption
}
```
