# Package: metaonly

Lightweight metadata-only object types for the garbage collector.

## Key Types

- `MetadataOnlyObject`: Allows decoding only apiVersion, kind, and metadata from JSON (not protobuf)
- `MetadataOnlyObjectList`: List type for MetadataOnlyObject items

## Purpose

Provides types for efficient partial deserialization of Kubernetes objects. The garbage collector only needs metadata (especially owner references) and can skip deserializing the full object spec/status.

## Design Notes

- Uses `+k8s:deepcopy-gen` annotations for code generation
- Implements `runtime.Object` interface
- Currently JSON-only; TODO to enable for protobuf
- Reduces memory and CPU usage when processing large numbers of objects
