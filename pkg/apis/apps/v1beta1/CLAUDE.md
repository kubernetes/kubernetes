# Package: v1beta1

## Purpose
Provides v1beta1 versioned API support for the apps API group, including type registration, defaulting, and conversion. This is a legacy beta version primarily for StatefulSet and Deployment.

## Key Constants/Variables
- `GroupName`: "apps"
- `SchemeGroupVersion`: apps/v1beta1

## Key Functions

### Defaulting (defaults.go)
- `SetDefaults_StatefulSet`: Sets OrderedReady pod management, OnDelete update strategy (differs from v1), auto-generates selector from template labels, Replicas=1, RevisionHistoryLimit=10
- `SetDefaults_Deployment`: Sets RollingUpdate with 25% MaxUnavailable/MaxSurge, RevisionHistoryLimit=2 (differs from v1's 10), ProgressDeadlineSeconds=600

### Conversion (conversion.go)
- `Convert_autoscaling_ScaleStatus_To_v1beta1_ScaleStatus`: Handles scale status with selector conversion
- `Convert_v1beta1_ScaleStatus_To_autoscaling_ScaleStatus`: Reverse conversion
- `Convert_v1beta1_StatefulSetSpec_To_apps_StatefulSetSpec`: Preserves VolumeClaimTemplate behavior
- `Convert_apps_StatefulSetSpec_To_v1beta1_StatefulSetSpec`: Sets APIVersion/Kind on VolumeClaimTemplates

## Key Differences from v1
- StatefulSet defaults to OnDelete strategy (v1 uses RollingUpdate)
- Deployment RevisionHistoryLimit defaults to 2 (v1 uses 10)
- Auto-generates selector from template labels if not specified

## Design Notes
- Uses field label conversion for StatefulSet metadata fields
- Supports both MatchLabels selector and TargetSelector for scale
- Maintains backward compatibility with pre-1.17 behavior for VolumeClaimTemplates
