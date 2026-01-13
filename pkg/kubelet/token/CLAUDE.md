# Package: token

## Purpose
The `token` package implements a manager for service account tokens for pods running on the node. It caches tokens and handles token refresh via the TokenRequest API.

## Key Types/Structs

- **Manager**: Manages service account tokens with caching. Contains a cache map, mutex for thread safety, token fetcher function, and clock for time operations.

## Key Functions

- **NewManager**: Creates a new token manager. Initializes the cache and starts a background goroutine for cache cleanup. Checks if the API server supports token requests.
- **GetServiceAccountToken**: Gets a token from cache or requests a new one via TokenRequest API. Implements refresh logic:
  - Returns cached token if not requiring refresh
  - Attempts to refresh if token is old
  - Falls back to cached token if refresh fails but token is still valid
  - Returns error if token is expired and refresh fails
- **DeleteServiceAccountToken**: Removes cached tokens for a deleted pod (by pod UID).
- **cleanup**: Background goroutine that runs every minute (gcPeriod) to remove expired tokens from cache.
- **requiresRefresh**: Returns true if token is older than 80% of its TTL or older than 24 hours (maxTTL).

## Constants

- **maxTTL**: 24 hours - maximum time before forced refresh
- **gcPeriod**: 1 minute - cleanup interval
- **maxJitter**: 10 seconds - random jitter added to refresh timing

## Design Notes

- Tokens are cached by a composite key of service account name, namespace, audiences, expiration, and bound object reference.
- Refresh is triggered proactively before token expires (at 80% of TTL).
- Graceful degradation: if refresh fails but token is still valid, logs error and returns cached token.
- Supports detection of API servers that don't have TokenRequest endpoint enabled.
- Thread-safe via RWMutex on cache access.
