# Package: scheduler

Provides rate-limited timed queue implementations for controlled node eviction scheduling.

## Key Types

- **TimedValue**: A value with timestamps for when it was added and when it should be processed.
- **TimedQueue**: Priority heap ordered by ProcessAt time (lowest first).
- **UniqueQueue**: FIFO queue that prevents duplicate entries until explicitly removed.
- **RateLimitedTimedQueue**: Combines UniqueQueue with a rate limiter for controlled processing.

## Key Functions

- **NewRateLimitedTimedQueue**: Creates a new rate-limited queue with the given limiter.
- **Try**: Processes queue items respecting rate limits; retries with specified delays.
- **Add**: Adds an item to be processed (deduplicated by value).
- **Remove**: Removes an item, allowing it to be re-added later.
- **SwapLimiter**: Dynamically changes the rate limit (used for zone state transitions).

## Key Constants

- **NodeHealthUpdateRetry**: 5 retries for node health updates.
- **NodeEvictionPeriod**: 100ms between eviction attempts.
- **EvictionRateLimiterBurst**: Burst value of 1 for rate limiters.

## Design Patterns

- Implements a priority queue using container/heap.
- Uses token bucket rate limiting for controlled eviction.
- Supports dynamic rate limit adjustment for zone-aware eviction control.
- Thread-safe with mutex protection.
