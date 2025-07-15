# goyaml.v2

This package provides type and function aliases for the `go.yaml.in/yaml/v2` package (which is compatible with `gopkg.in/yaml.v2`).

## Purpose

The purpose of this package is to:

1. Provide a transition path for users migrating from the sigs.k8s.io/yaml package to direct usage of go.yaml.in/yaml/v2
2. Maintain compatibility with existing code while encouraging migration to the upstream package
3. Reduce maintenance overhead by delegating to the upstream implementation

## Usage

Instead of importing this package directly, you should migrate to using `go.yaml.in/yaml/v2` directly:

```go
// Old way
import "sigs.k8s.io/yaml/goyaml.v2"

// Recommended way
import "go.yaml.in/yaml/v2"
```

## Available Types and Functions

All public types and functions from `go.yaml.in/yaml/v2` are available through this package:

### Types

- `MapSlice` - Encodes and decodes as a YAML map with preserved key order
- `MapItem` - An item in a MapSlice
- `Unmarshaler` - Interface for custom unmarshaling behavior
- `Marshaler` - Interface for custom marshaling behavior
- `IsZeroer` - Interface to check if an object is zero
- `Decoder` - Reads and decodes YAML values from an input stream
- `Encoder` - Writes YAML values to an output stream
- `TypeError` - Error returned by Unmarshal for decoding issues

### Functions

- `Unmarshal` - Decodes YAML data into a Go value
- `UnmarshalStrict` - Like Unmarshal but errors on unknown fields
- `Marshal` - Serializes a Go value into YAML
- `NewDecoder` - Creates a new Decoder
- `NewEncoder` - Creates a new Encoder
- `FutureLineWrap` - Controls line wrapping behavior

## Migration Guide

To migrate from this package to `go.yaml.in/yaml/v2`:

1. Update your import statements:
   ```go
   // From
   import "sigs.k8s.io/yaml/goyaml.v2"
   
   // To
   import "go.yaml.in/yaml/v2"
   ```

2. No code changes should be necessary as the API is identical

3. Update your go.mod file to include the dependency:
   ```
   require go.yaml.in/yaml/v2 v2.4.2
   ```

## Deprecation Notice

All types and functions in this package are marked as deprecated. You should migrate to using `go.yaml.in/yaml/v2` directly.
