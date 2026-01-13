# Package testing

Package testing provides mock implementations for CPU manager state testing.

## Key Types

- `MockCheckpoint`: Simple mock implementation of the Checkpoint interface

## MockCheckpoint

Fields:
- `Content`: String content to return from marshal

Methods:
- `MarshalCheckpoint()`: Returns Content as bytes
- `UnmarshalCheckpoint(blob)`: No-op, always succeeds
- `VerifyChecksum()`: No-op, always succeeds

## Design Notes

- Implements checkpointmanager.Checkpoint interface
- Used for unit testing checkpoint-related functionality
- Does not perform actual serialization or checksum verification
