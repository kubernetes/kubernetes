# Package: horizontalpodautoscaler

## Purpose
Implements the registry strategy for HorizontalPodAutoscaler (HPA) resources, which automatically scale workloads based on observed metrics.

## Key Types

- **autoscalerStrategy**: Implements verification logic and REST strategies for HPAs
- **autoscalerStatusStrategy**: Strategy for status-only updates

## Key Functions

- **Strategy**: Default logic for creating/updating HPA objects
- **StatusStrategy**: Default logic for status updates
- **PrepareForCreate()**: Clears status, drops disabled fields
- **PrepareForUpdate()**: Preserves status, drops disabled fields
- **Validate()**: Validates new HPAs including declarative validation
- **ValidateUpdate()**: Validates HPA updates with migration checks
- **GetResetFields()**: Returns fields to reset for autoscaling/v1 and autoscaling/v2
- **dropDisabledFields()**: Drops Tolerance field if HPAConfigurableTolerance feature is disabled
- **validationOptionsForHorizontalPodAutoscaler()**: Configures validation options based on feature gates

## Design Notes

- Namespace-scoped resource
- Supports HPAScaleToZero feature gate (allows minReplicas=0)
- Supports HPAConfigurableTolerance feature gate (allows custom scaling tolerance)
- Handles validation for both autoscaling/v1 and autoscaling/v2 API versions
- Special handling for ReplicationController (allows empty apiVersion in scaleTargetRef)
- Uses declarative validation with migration checks
