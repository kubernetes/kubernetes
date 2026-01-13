# Package: config

Contains configuration types for the Horizontal Pod Autoscaler (HPA) controller.

## Key Types

- **HPAControllerConfiguration**: Configuration struct containing:
  - `ConcurrentHorizontalPodAutoscalerSyncs`: Number of HPAs synced concurrently.
  - `HorizontalPodAutoscalerSyncPeriod`: How often HPAs are reconciled.
  - `HorizontalPodAutoscalerDownscaleStabilizationWindow`: Lookback period for scale-down decisions.
  - `HorizontalPodAutoscalerTolerance`: Threshold for triggering scale changes.
  - `HorizontalPodAutoscalerCPUInitializationPeriod`: Time after pod start to skip CPU samples.
  - `HorizontalPodAutoscalerInitialReadinessDelay`: Time to ignore readiness changes after pod start.

## Design Patterns

- Part of the componentconfig pattern for kube-controller-manager.
- Tolerance prevents scaling oscillation when metrics are near target values.
- Initialization periods handle pod startup behavior where metrics may be unreliable.
