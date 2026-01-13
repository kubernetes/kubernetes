# Package v1

Package v1 provides example checkpoint data structures for testing checkpoint format compatibility and versioning.

## Key Types

- `CheckpointData`: Sample checkpoint implementation with version, name, data, and checksum
- `Data`: Container data structure with port mappings and host network flag
- `PortMapping`: Port mapping configuration (protocol, container port, host port)

## Checkpoint Implementation

`CheckpointData` implements the Checkpoint interface:
- `MarshalCheckpoint()`: JSON marshals the checkpoint, computing checksum from Data
- `UnmarshalCheckpoint(blob)`: JSON unmarshals the checkpoint
- `VerifyChecksum()`: Validates the checksum against the Data field

## Design Notes

- Serves as a reference implementation for checkpoint format
- Used in tests to verify checkpoint manager functionality
- Demonstrates proper checksum calculation over data fields only (not the checksum itself)
- Models real-world checkpoint use cases like pod sandbox checkpoints
