# Package: storage

## Purpose
Provides the REST storage implementation for LimitRange objects, wrapping the generic registry store.

## Key Types

- **REST**: Embeds `genericregistry.Store` to provide RESTful storage operations for LimitRanges.

## Key Functions

- **NewREST(optsGetter)**: Creates and returns a configured REST storage object for LimitRanges. Sets up:
  - Object creation functions for LimitRange and LimitRangeList
  - Create/Update/Delete strategies from the limitrange package
  - Default table converter (TODO note indicates custom converter needed)

- **ShortNames()**: Returns `["limits"]` - the kubectl short name for limitranges.

## Design Notes

- Implements `rest.ShortNamesProvider` interface.
- Uses `rest.NewDefaultTableConvertor` instead of custom table generation (noted as TODO for improvement).
- Standard generic registry pattern.
