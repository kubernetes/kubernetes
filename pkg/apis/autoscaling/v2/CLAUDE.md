# Package: v2

## Purpose
Provides v2 versioned API support for the autoscaling API group with full HPA functionality, including multiple metrics, scaling behavior, and tolerance configuration.

## Key Constants/Variables
- `SchemeGroupVersion`: autoscaling/v2

## Default Values
### Scale Up Defaults
- `scaleUpLimitPercent`: 100 (double pods)
- `scaleUpLimitMinimumPods`: 4
- `scaleUpPeriod`: 15 seconds
- `scaleUpStabilizationSeconds`: 0 (no stabilization)
- `selectPolicy`: Max

### Scale Down Defaults
- `scaleDownLimitPercent`: 100 (can remove all)
- `scaleDownPeriod`: 15 seconds
- `stabilizationWindowSeconds`: nil (uses controller default of 300s)
- `selectPolicy`: Max

## Key Functions

### Defaulting (defaults.go)
- `SetDefaults_HorizontalPodAutoscaler`: Sets MinReplicas=1, default CPU metric (80% utilization) if no metrics specified, applies behavior defaults
- `SetDefaults_HorizontalPodAutoscalerBehavior`: Fills unset behavior fields with defaults
- `GenerateHPAScaleUpRules`: Returns fully-initialized scale-up rules
- `GenerateHPAScaleDownRules`: Returns fully-initialized scale-down rules
- `copyHPAScalingRules`: Copies non-nil fields from source to default rules

## Design Notes
- v2 is the full-featured HPA API version
- Supports Object, Pods, Resource, ContainerResource, and External metrics
- Behavior configuration allows fine-grained control of scaling velocity
- Tolerance feature (HPAConfigurableTolerance) prevents scaling for small variations
- Default CPU utilization is 80% (defined in autoscaling.DefaultCPUUtilization)
