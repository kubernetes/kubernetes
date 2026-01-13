# Package: validation

## Purpose
Provides comprehensive validation logic for autoscaling API types including HorizontalPodAutoscaler and Scale resources.

## Key Constants
- `MaxPeriodSeconds`: 1800 (30 minutes) - maximum scaling policy period
- `MaxStabilizationWindowSeconds`: 3600 (1 hour) - maximum stabilization window

## Key Types
- `HorizontalPodAutoscalerSpecValidationOptions`: MinReplicasLowerBound, ScaleTargetRefValidationOptions, ObjectMetricsValidationOptions
- `CrossVersionObjectReferenceValidationOptions`: AllowEmptyAPIGroup, AllowInvalidAPIVersion

## Key Validation Functions

### HPA Validation
- `ValidateHorizontalPodAutoscaler`: Full validation for create
- `ValidateHorizontalPodAutoscalerUpdate`: Validates updates
- `ValidateHorizontalPodAutoscalerStatusUpdate`: Validates status updates
- `validateHorizontalPodAutoscalerSpec`: Validates spec including min/max replicas, scale target ref, metrics, and behavior

### Scale Validation
- `ValidateScale`: Validates Scale resource (replicas must be non-negative)

### Reference Validation
- `ValidateCrossVersionObjectReference`: Validates Kind, Name, and APIVersion
- `ValidateAPIVersion`: Validates APIVersion format and group

### Metrics Validation
- `validateMetrics`: Validates metrics array, requires Object or External metric for scale-to-zero
- `validateMetricSpec`: Validates individual metric specs (Object, Pods, Resource, ContainerResource, External)
- `validateMetricTarget`: Validates target type (Utilization, Value, AverageValue) and positive values
- `validateMetricIdentifier`: Validates metric name

### Behavior Validation
- `validateBehavior`: Validates ScaleUp and ScaleDown rules
- `validateScalingRules`: Validates stabilization window, select policy, policies list, and tolerance
- `validateScalingPolicy`: Validates policy type (Pods/Percent), value (>0), and period (1-1800s)

## Validation Rules
- MinReplicas must be >= configured lower bound (usually 0 or 1)
- MaxReplicas must be > 0 and >= MinReplicas
- Scale-to-zero requires at least one Object or External metric
- Exactly one metric source field must match the type
- Select policy must be Max, Min, or Disabled
- Stabilization window must be 0-3600 seconds
- Policy period must be 1-1800 seconds
- Tolerance must be non-negative
