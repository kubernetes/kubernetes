# Package: v1

## Purpose
Provides v1 versioned API support for the autoscaling API group, including type registration, defaulting, and conversion between v1 and internal types.

## Key Constants/Variables
- `GroupName`: "autoscaling"
- `SchemeGroupVersion`: autoscaling/v1

## Key Functions

### Defaulting (defaults.go)
- `SetDefaults_HorizontalPodAutoscaler`: Sets MinReplicas=1 if not specified

### Conversion (conversion.go)
- Handles conversion between v1 and internal HPA types
- v1 HPA supports only CPU utilization via annotation for additional metrics
- More advanced metrics require v2

## Design Notes
- v1 is a stable but limited version
- Only supports targetCPUUtilizationPercentage natively
- Additional metrics stored in annotations for backward compatibility
- CPU utilization default applied in conversion (not defaulting) due to annotation access requirements
