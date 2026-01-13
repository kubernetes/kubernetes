# Package: cache

## Purpose
The `cache` package provides a simple TTL-based object cache with automatic refresh capability.

## Key Types/Structs

- **ObjectCache**: Wrapper around client-go's TTL expiration cache with string keys and an updater function for automatic refresh on cache miss.
- **objectEntry**: Internal struct holding the key-value pair for cache entries.

## Key Functions

- **NewObjectCache**: Creates a new ObjectCache with a custom updater function and TTL duration.
- **Get**: Retrieves an object by key. If not found or expired, calls the updater function to refresh and caches the result.
- **Add**: Manually adds an object to the cache with the given key.

## Design Notes

- Uses client-go's TTLStore for expiration-based caching.
- The updater function is called automatically on cache miss, making the cache self-refreshing.
- Thread-safe via the underlying TTLStore implementation.
- Useful for caching expensive-to-compute values that can be reused within a TTL window.
