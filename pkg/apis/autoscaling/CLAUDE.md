# Package: autoscaling

## Purpose
Defines the internal (unversioned) API types for the autoscaling API group, providing HorizontalPodAutoscaler (HPA) and Scale resources for automatic pod scaling.

## Key Types

### HorizontalPodAutoscaler
Automatically manages replica count based on observed metrics.
- `HorizontalPodAutoscalerSpec`: ScaleTargetRef, MinReplicas, MaxReplicas, Metrics, Behavior
- `HorizontalPodAutoscalerStatus`: CurrentReplicas, DesiredReplicas, CurrentMetrics, Conditions, LastScaleTime

### Scale
Generic scaling subresource for any scalable resource.
- `ScaleSpec`: Replicas (desired count)
- `ScaleStatus`: Replicas (actual count), Selector

### Metrics Configuration
- `MetricSpec`: Configures scaling based on Object, Pods, Resource, ContainerResource, or External metrics
- `MetricTarget`: Specifies target as Value, AverageValue, or AverageUtilization
- `MetricIdentifier`: Name and optional label selector for a metric

### Scaling Behavior
- `HorizontalPodAutoscalerBehavior`: Configures ScaleUp and ScaleDown rules
- `HPAScalingRules`: StabilizationWindowSeconds, SelectPolicy (Max/Min/Disabled), Policies, Tolerance
- `HPAScalingPolicy`: Type (Pods/Percent), Value, PeriodSeconds

## Key Constants
- `MetricSourceType`: Object, Pods, Resource, External, ContainerResource
- `MetricTargetType`: Utilization, Value, AverageValue
- `ScalingPolicySelect`: Max, Min, Disabled
- `HPAScalingPolicyType`: Pods, Percent

## Key Functions
- `AddToScheme`: Registers Scale, HorizontalPodAutoscaler, and HorizontalPodAutoscalerList
- `Kind(kind string)`: Returns Group-qualified GroupKind
- `Resource(resource string)`: Returns Group-qualified GroupResource

## Design Notes
- Supports scaling to zero with Object or External metrics (requires feature gate)
- Behavior configuration allows fine-grained control over scaling velocity
- Tolerance feature (beta) prevents scaling for small metric variations
