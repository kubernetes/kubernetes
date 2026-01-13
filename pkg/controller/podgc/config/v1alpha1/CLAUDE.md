# Package: v1alpha1

Provides versioned defaults for Pod GC controller configuration.

## Key Functions

- **RecommendedDefaultPodGCControllerConfiguration**: Sets recommended defaults:
  - `TerminatedPodGCThreshold`: 12500

## Design Patterns

- Explicit defaulting function pattern.
- The 12500 threshold allows significant terminated pod history while preventing unbounded growth.
