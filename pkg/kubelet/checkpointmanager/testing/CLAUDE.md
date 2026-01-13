# Package testing

Package testing provides in-memory store implementation for testing checkpoint manager functionality.

## Key Types

- `MemStore`: Thread-safe in-memory implementation of the checkpoint store interface

## Key Functions

- `NewMemStore()`: Creates a new in-memory store
- `Write(key, data)`: Stores data under the given key
- `Read(key)`: Retrieves data for the given key
- `Delete(key)`: Removes data for the given key
- `List()`: Returns all stored keys

## Design Notes

- Uses map[string][]byte for storage
- Thread-safe via mutex
- Returns "checkpoint is not found" error on missing keys
- Useful for unit tests that need checkpoint persistence without filesystem
