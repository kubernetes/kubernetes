# goyaml.v3

This package provides type and function aliases for the `go.yaml.in/yaml/v3` package (which is compatible with `gopkg.in/yaml.v3`).

## Purpose

The purpose of this package is to:

1. Provide a transition path for users migrating from the sigs.k8s.io/yaml package to direct usage of go.yaml.in/yaml/v3
2. Maintain compatibility with existing code while encouraging migration to the upstream package
3. Reduce maintenance overhead by delegating to the upstream implementation

## Usage

Instead of importing this package directly, you should migrate to using `go.yaml.in/yaml/v3` directly:

```go
// Old way
import "sigs.k8s.io/yaml/goyaml.v3"

// Recommended way
import "go.yaml.in/yaml/v3"
```

## Available Types and Functions

All public types and functions from `go.yaml.in/yaml/v3` are available through this package:

### Types

- `Unmarshaler` - Interface for custom unmarshaling behavior
- `Marshaler` - Interface for custom marshaling behavior
- `IsZeroer` - Interface to check if an object is zero
- `Decoder` - Reads and decodes YAML values from an input stream
- `Encoder` - Writes YAML values to an output stream
- `TypeError` - Error returned by Unmarshal for decoding issues
- `Node` - Represents a YAML node in the document
- `Kind` - Represents the kind of a YAML node
- `Style` - Represents the style of a YAML node

### Functions

- `Unmarshal` - Decodes YAML data into a Go value
- `Marshal` - Serializes a Go value into YAML
- `NewDecoder` - Creates a new Decoder
- `NewEncoder` - Creates a new Encoder

## Migration Guide

To migrate from this package to `go.yaml.in/yaml/v3`:

1. Update your import statements:
   ```go
   // From
   import "sigs.k8s.io/yaml/goyaml.v3"
   
   // To
   import "go.yaml.in/yaml/v3"
   ```

2. No code changes should be necessary as the API is identical

3. Update your go.mod file to include the dependency:
   ```
   require go.yaml.in/yaml/v3 v3.0.3
   ```

## Deprecation Notice

All types and functions in this package are marked as deprecated. You should migrate to using `go.yaml.in/yaml/v3` directly.
