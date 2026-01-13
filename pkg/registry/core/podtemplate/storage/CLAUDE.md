# Package: storage

## Purpose
Provides REST storage implementation for PodTemplate objects.

## Key Types

- **REST**: Main storage for PodTemplate operations using genericregistry.Store.

## Key Functions

- **NewREST(optsGetter)**: Returns REST instance configured with:
  - ReturnDeletedObject: true
  - Uses podtemplate.MatchPodTemplate predicate

## Design Notes

- Simple storage implementation with no subresources.
- Returns deleted objects on delete operations.
- Uses standard genericregistry.Store pattern.
