# Package: configfiles

Loads KubeletConfiguration from filesystem configuration files.

## Key Types/Structs

- **Loader**: Interface for loading KubeletConfiguration from storage.
- **fsLoader**: Implementation that loads configuration from a file on the filesystem.

## Key Functions

- `NewFsLoader(fs, kubeletFile)`: Creates a Loader that reads KubeletConfiguration from the specified file path.
- `Load()`: Reads and decodes the kubelet config file, resolving relative paths to absolute paths.
- `LoadIntoJSON()`: Reads and decodes the config file, returning it as JSON bytes with GVK.
- `resolveRelativePaths()`: Converts relative paths in the config to absolute paths based on the config file's directory.

## Design Notes

- Uses the kubelet scheme and codecs for proper serialization/deserialization
- Empty configuration files are treated as errors
- Relative paths in the configuration are resolved relative to the config file's directory
- Supports strict decoding for configuration validation
