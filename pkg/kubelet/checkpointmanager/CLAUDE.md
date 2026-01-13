# Package checkpointmanager

Package checkpointmanager provides a generic interface for persisting and retrieving checkpoint data to survive kubelet restarts.

## Key Types

- `Checkpoint`: Interface that checkpoint data must implement (marshal, unmarshal, verify checksum)
- `CheckpointManager`: Interface for CRUD operations on checkpoints
- `impl`: File-backed implementation using utilstore

## Key Functions

- `NewCheckpointManager(checkpointDir)`: Creates a new file-backed checkpoint manager
- `CreateCheckpoint`: Marshals and persists checkpoint data
- `GetCheckpoint`: Retrieves and unmarshals checkpoint, verifying checksum
- `RemoveCheckpoint`: Deletes a checkpoint (no error if not found)
- `ListCheckpoints`: Returns all checkpoint keys

## Checkpoint Interface Requirements

Checkpoint implementations must provide:
- `MarshalCheckpoint() ([]byte, error)`: Serialize to bytes
- `UnmarshalCheckpoint(blob []byte) error`: Deserialize from bytes
- `VerifyChecksum() error`: Validate data integrity

## Design Notes

- Thread-safe via mutex
- Uses file-based store for persistence
- Checksum verification on read prevents corruption
- Used by device manager, pod sandbox, and other kubelet components
