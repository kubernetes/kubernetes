# Package: endpointslice

## Purpose
Provides utility types for EndpointSlice controllers to handle stale informer cache errors.

## Key Types

- **StaleInformerCache**: Error type indicating that the informer cache contains out-of-date resources.

## Key Functions

- **NewStaleInformerCache(msg)**: Creates a new StaleInformerCache error with the given message.
- **Error()**: Returns the error message string (implements error interface).
- **IsStaleInformerCacheErr(err)**: Type checks if an error is a StaleInformerCache error.

## Design Notes

- Used by EndpointSlice controllers to distinguish cache staleness errors from other error types.
- Allows controllers to handle stale cache situations differently (e.g., retry vs fail).
