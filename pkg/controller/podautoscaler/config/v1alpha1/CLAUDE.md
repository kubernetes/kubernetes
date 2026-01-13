# Package: v1alpha1

Provides versioned defaults for HPA controller configuration.

## Key Functions

- **RecommendedDefaultHPAControllerConfiguration**: Sets recommended defaults:
  - `ConcurrentHorizontalPodAutoscalerSyncs`: 5
  - `HorizontalPodAutoscalerSyncPeriod`: 15 seconds
  - `HorizontalPodAutoscalerDownscaleStabilizationWindow`: 5 minutes
  - `HorizontalPodAutoscalerCPUInitializationPeriod`: 5 minutes
  - `HorizontalPodAutoscalerInitialReadinessDelay`: 30 seconds
  - `HorizontalPodAutoscalerTolerance`: 0.1 (10%)

## Design Patterns

- Explicit defaulting function pattern allowing consumers to opt-out.
- The 5-minute downscale stabilization prevents rapid scale-down fluctuations.
- 10% tolerance prevents scaling when metrics are within acceptable range of target.
