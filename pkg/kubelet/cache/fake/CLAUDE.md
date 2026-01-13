# Package fake

Package fake provides a mock implementation of the scheduler cache for unit testing.

## Key Types

- `Cache`: Embeds the real cache interface with optional mock function overrides

## Mock Function Fields

- `AssumeFunc`: Override for AssumePod behavior
- `ForgetFunc`: Override for ForgetPod behavior
- `IsAssumedPodFunc`: Override for IsAssumedPod behavior
- `GetPodFunc`: Override for GetPod behavior

## Key Methods

- `AssumePod`: Calls AssumeFunc if set, otherwise delegates to embedded cache
- `ForgetPod`: Calls ForgetFunc if set, otherwise delegates to embedded cache
- `IsAssumedPod`: Calls IsAssumedPodFunc if set, otherwise delegates to embedded cache
- `GetPod`: Calls GetPodFunc if set, otherwise delegates to embedded cache

## Design Notes

- Allows selective mocking of specific cache methods
- Unmocked methods delegate to the real cache implementation
- Useful for testing scheduler components that depend on cache behavior
