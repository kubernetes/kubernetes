# Package: exponentialbackoff

## Purpose
Implements exponential backoff logic for rate-limiting retries after operation failures.

## Key Types
- `ExponentialBackoff` - Tracks last error, time, and current backoff duration

## Key Functions
- `SafeToRetry()` - Returns error if still in backoff period, nil if safe to retry
- `Update()` - Updates backoff state after an error (doubles duration)
- `GenerateNoRetriesPermittedMsg()` - Creates human-readable message about backoff state
- `NewExponentialBackoffError()` - Creates a backoff error for a named operation
- `IsExponentialBackoff()` - Checks if an error is an exponential backoff error

## Constants
- Initial backoff: 500ms
- Maximum backoff: ~2 minutes (2m2s)

## Design Patterns
- Backoff duration doubles after each error
- Capped at maximum to prevent unbounded waits
- Timestamps used to determine if backoff period has elapsed
- Integrates with goroutinemap for operation-level backoff
