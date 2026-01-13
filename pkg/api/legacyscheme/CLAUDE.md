# Package: legacyscheme

## Purpose
Provides the default runtime.Scheme instance for the Kubernetes API where all core API types are registered. This is a singleton scheme used throughout the API server.

## Key Variables
- `Scheme` - The default `runtime.Scheme` instance with all Kubernetes API types registered
- `Codecs` - `serializer.CodecFactory` for encoding/decoding API objects using the Scheme
- `ParameterCodec` - `runtime.ParameterCodec` for handling query parameter versioning

## Important Notes
- This Scheme is special and should only appear in the core api group
- When creating new API groups, copy from the extensions group instead, not this package
- The Scheme is the central registry for type conversions, defaulting, and serialization
