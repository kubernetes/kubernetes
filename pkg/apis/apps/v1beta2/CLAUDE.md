# Package: v1beta2

## Purpose
Provides v1beta2 versioned API support for the apps API group, including type registration, defaulting, and conversion. This version is closer to v1 than v1beta1.

## Key Constants/Variables
- `GroupName`: "apps"
- `SchemeGroupVersion`: apps/v1beta2

## Key Functions

### Defaulting (defaults.go)
- `SetDefaults_DaemonSet`: RollingUpdate with MaxUnavailable=1, MaxSurge=0, RevisionHistoryLimit=10
- `SetDefaults_StatefulSet`: OrderedReady pod management, RollingUpdate strategy with Partition=0, Replicas=1, RevisionHistoryLimit=10, Retain PVC policy
- `SetDefaults_Deployment`: RollingUpdate with 25% MaxUnavailable/MaxSurge, Replicas=1, RevisionHistoryLimit=10, ProgressDeadlineSeconds=600
- `SetDefaults_ReplicaSet`: Replicas=1

### Conversion (conversion.go)
- `Convert_autoscaling_ScaleStatus_To_v1beta2_ScaleStatus`: Scale status with selector handling
- `Convert_v1beta2_Deployment_To_apps_Deployment`: Handles deprecated RollbackTo annotation
- `Convert_apps_Deployment_To_v1beta2_Deployment`: Preserves RollbackTo via annotation
- `Convert_v1beta2_DaemonSet_To_apps_DaemonSet`: Handles TemplateGeneration annotation
- VolumeClaimTemplate conversion for StatefulSet (preserves APIVersion/Kind)

## Key Similarities to v1
- Same defaults for most resources (unlike v1beta1)
- Uses RollingUpdate as default StatefulSet strategy
- Same RevisionHistoryLimit defaults

## Design Notes
- Bridge version between v1beta1 and v1
- Supports MaxUnavailableStatefulSet feature gate
- Uses annotation-based conversion for deprecated fields
