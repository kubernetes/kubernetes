# Package: fake

## Purpose
Provides a fake implementation of the scheduler cache for use in unit tests. Allows mocking specific cache methods while delegating others to the embedded real cache.

See `../CLAUDE.md` for details on the cache architecture (pod state machine, optimistic scheduling, snapshot mechanism).

## Key Types

### Cache
Fake cache struct embedding the real cache interface:
```go
type Cache struct {
    internalcache.Cache
    AssumeFunc       func(*v1.Pod)
    ForgetFunc       func(*v1.Pod)
    IsAssumedPodFunc func(*v1.Pod) bool
    GetPodFunc       func(*v1.Pod) *v1.Pod
}
```

## Mockable Methods

- **AssumePod**: If AssumeFunc is set, calls it; otherwise delegates to real cache
- **ForgetPod**: If ForgetFunc is set, calls it; otherwise delegates to real cache
- **IsAssumedPod**: If IsAssumedPodFunc is set, calls it; otherwise delegates to real cache
- **GetPod**: If GetPodFunc is set, calls it; otherwise delegates to real cache

## Usage Example
```go
fakeCache := &fake.Cache{
    Cache: realCache,
    AssumeFunc: func(pod *v1.Pod) {
        // Custom assume behavior for test
    },
}
```

## Design Pattern
- Decorator pattern: wraps real cache with optional overrides
- Selective mocking: only mock methods you need to control
- Falls back to real implementation for unmocked methods
- Commonly used in scheduler unit tests
