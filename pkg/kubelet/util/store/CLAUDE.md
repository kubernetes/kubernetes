# Package: store

## Purpose
The `store` package defines a thread-safe key-value store interface and provides a file-based implementation for persisting data to disk.

## Key Interfaces

- **Store**: Thread-safe key-value storage interface.
  - `Write(key, data)`: Writes data with key.
  - `Read(key)`: Retrieves data by key (returns ErrKeyNotFound if not found).
  - `Delete(key)`: Deletes data by key (no error if key doesn't exist).
  - `List()`: Lists all existing keys.

## Key Functions

- **ValidateKey**: Validates that a key meets format and length requirements.

## Key Validation Rules

- Maximum length: 250 characters
- Must start and end with alphanumeric character
- May contain: `A-Za-z0-9`, `-`, `_`, `.`
- Regex: `^([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]$`

## Constants/Errors

- **keyMaxLength**: 250
- **ErrKeyNotFound**: Returned when a key is not found in the store.

## Implementation

The package includes a file-based implementation (filestore.go) that:
- Stores each key as a separate file in a directory
- Uses atomic writes for data integrity
- Is thread-safe for concurrent access

## Design Notes

- Used by kubelet components to persist state across restarts.
- Examples: user namespace mappings, checkpoint data.
- Key format matches Kubernetes naming conventions.
