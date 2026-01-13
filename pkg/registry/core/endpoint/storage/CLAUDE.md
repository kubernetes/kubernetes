# Package: storage

## Purpose
Provides the REST storage implementation for Endpoints objects, wrapping the generic registry store with Endpoints-specific configuration.

## Key Types

- **REST**: Embeds `genericregistry.Store` to provide RESTful storage operations for Endpoints.

## Key Functions

- **NewREST(optsGetter)**: Creates and returns a configured REST storage object for Endpoints. Sets up:
  - Object creation functions for Endpoints and EndpointsList
  - Create/Update/Delete strategies from the endpoint package
  - Table conversion for kubectl output

- **ShortNames()**: Returns `["ep"]` - the kubectl short name for endpoints.

## Design Notes

- Implements `rest.ShortNamesProvider` interface.
- The singular and plural qualified resource names are both "endpoints" (grammatically plural).
- Uses standard generic registry pattern from apiserver.
