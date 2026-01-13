# Package: v1

## Purpose
Provides v1 versioned API support for the apps API group, including type registration, defaulting, and conversion between v1 and internal types.

## Key Constants/Variables
- `GroupName`: "apps"
- `SchemeGroupVersion`: apps/v1

## Key Functions

### Defaulting (defaults.go)
- `SetDefaults_Deployment`: Sets Replicas=1, RollingUpdate strategy with 25% MaxUnavailable/MaxSurge, RevisionHistoryLimit=10, ProgressDeadlineSeconds=600
- `SetDefaults_DaemonSet`: Sets RollingUpdate strategy with MaxUnavailable=1, MaxSurge=0, RevisionHistoryLimit=10
- `SetDefaults_StatefulSet`: Sets OrderedReady pod management, RollingUpdate strategy, Partition=0, Replicas=1, RevisionHistoryLimit=10, Retain PVC policy
- `SetDefaults_ReplicaSet`: Sets Replicas=1

### Conversion (conversion.go)
- Handles Deployment rollback annotation roundtrip (deprecated RollbackTo field)
- Handles DaemonSet template generation annotation conversion
- Preserves StatefulSet VolumeClaimTemplate APIVersion/Kind for backward compatibility

## Design Notes
- v1 is the stable API version for apps
- Defaults differ from extensions (25% vs 1 for rolling update values)
- Uses feature gates for MaxUnavailableStatefulSet
- Conversion functions handle deprecated field migrations via annotations
