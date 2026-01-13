# Package: scheme

## Purpose
Provides the runtime scheme and codec factory for scheduler configuration types. This scheme is used for serializing and deserializing scheduler configuration.

## Key Variables

- **Scheme**: Runtime scheme with all kubescheduler API types registered.
- **Codecs**: Codec factory for encoding/decoding scheduler configuration (strict mode enabled).

## Key Functions

- **AddToScheme(scheme)**: Registers all scheduler config types to a scheme:
  - Internal types from config package
  - Versioned types from configv1 package (v1)
  - Sets v1 as the preferred version

## Design Notes

- Uses serializer.EnableStrict for strict JSON/YAML parsing.
- Initialized via init() function for package-level availability.
- Used when loading scheduler configuration from files.
- Supports conversion between internal and versioned types.
