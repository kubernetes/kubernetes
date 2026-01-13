# Package: feature

## Purpose
Provides a struct for passing feature gate values to scheduler plugins. This decouples plugins from the internal Kubernetes feature gate package, making them more testable and portable.

## Key Types

### Features
Struct containing boolean flags for all scheduler-relevant feature gates:

**DRA (Dynamic Resource Allocation) Features:**
- EnableDynamicResourceAllocation
- EnableDRAExtendedResource
- EnableDRAPrioritizedList
- EnableDRAAdminAccess
- EnableDRAConsumableCapacity
- EnableDRADeviceTaints
- EnableDRADeviceBindingConditions
- EnableDRAPartitionableDevices
- EnableDRAResourceClaimDeviceStatus
- EnableDRASchedulerFilterTimeout

**Pod Scheduling Features:**
- EnablePodLevelResources
- EnableInPlacePodVerticalScaling
- EnableInPlacePodLevelResourcesVerticalScaling
- EnableSidecarContainers
- EnableGangScheduling

**Node Features:**
- EnableNodeInclusionPolicyInPodTopologySpread
- EnableMatchLabelKeysInPodTopologySpread
- EnableNodeDeclaredFeatures
- EnableTaintTolerationComparisonOperators

**Storage Features:**
- EnableVolumeAttributesClass
- EnableCSIMigrationPortworx
- EnableVolumeLimitScaling
- EnableStorageCapacityScoring

**Scheduler Infrastructure:**
- EnableSchedulingQueueHint
- EnableAsyncPreemption

## Key Functions

- **NewSchedulerFeaturesFromGates(featureGate)**: Creates a Features struct by reading current feature gate values from the provided FeatureGate interface

## Design Pattern
- Snapshot pattern: captures feature gate state at scheduler initialization
- Dependency injection: plugins receive Features instead of directly accessing global feature gates
- Enables unit testing with custom feature configurations
