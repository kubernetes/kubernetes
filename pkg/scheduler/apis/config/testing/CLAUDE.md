# Package: testing

## Purpose
Provides test utilities for working with scheduler configuration types. Helps convert between versioned (v1) and internal configuration types with proper defaulting.

## Key Functions

### V1ToInternalWithDefaults
```go
func V1ToInternalWithDefaults(t *testing.T, versionedCfg v1.KubeSchedulerConfiguration) *config.KubeSchedulerConfiguration
```
Converts a v1 KubeSchedulerConfiguration to the internal type with defaults applied:
1. Sets recommended debugging configuration
2. Applies scheme defaults
3. Converts to internal type
4. Fails test on any error

## Usage Example
```go
func TestScheduler(t *testing.T) {
    cfg := testing.V1ToInternalWithDefaults(t, v1.KubeSchedulerConfiguration{
        Profiles: []v1.KubeSchedulerProfile{
            {SchedulerName: "custom-scheduler"},
        },
    })
    // Use cfg...
}
```

## Design Pattern
- Simplifies test setup by handling conversion and defaulting
- Uses testing.T for automatic failure on errors
- Ensures tests use properly defaulted configurations
