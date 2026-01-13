# Package: storage

## Purpose
Provides the REST storage implementation for ConfigMap objects, wrapping the generic registry store with ConfigMap-specific configuration.

## Key Types

- **REST**: Embeds `genericregistry.Store` to provide RESTful storage operations for ConfigMaps.

## Key Functions

- **NewREST(optsGetter)**: Creates and returns a configured REST storage object for ConfigMaps. Sets up:
  - Object creation functions for ConfigMap and ConfigMapList
  - Create/Update/Delete strategies from the configmap package
  - Table conversion for kubectl output
  - Short name "cm" for the resource

- **ShortNames()**: Returns `["cm"]` - the kubectl short name for configmaps.

## Design Notes

- Implements `rest.ShortNamesProvider` interface.
- Uses the standard generic registry pattern from apiserver.
- Delegates validation and preparation logic to the configmap.Strategy.
