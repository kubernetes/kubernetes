# Package: storage

## Purpose
Provides the REST storage implementation for Event objects with TTL support, wrapping the generic registry store.

## Key Types

- **REST**: Embeds `genericregistry.Store` to provide RESTful storage operations for Events.

## Key Functions

- **NewREST(optsGetter, ttl)**: Creates a configured REST storage object for Events. Notable features:
  - Accepts a `ttl` parameter for automatic event expiration
  - Sets up TTLFunc to apply the configured TTL to all events
  - Uses event.Matcher predicate for filtering
  - Configures table conversion for kubectl output

- **ShortNames()**: Returns `["ev"]` - the kubectl short name for events.

## Design Notes

- Implements `rest.ShortNamesProvider` interface.
- Events have a configurable TTL (time-to-live) for automatic cleanup, distinguishing them from most other resources.
- Uses event.GetAttrs for attribute-based filtering.
