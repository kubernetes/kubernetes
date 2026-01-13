# Package: validation

## Purpose
Provides comprehensive validation logic for all apps API types including StatefulSet, Deployment, DaemonSet, ReplicaSet, and ControllerRevision.

## Key Types
- `StatefulSetValidationOptions`: Controls validation behavior (AllowInvalidServiceName, SkipValidatePodTemplateSpec, SkipValidateVolumeClaimTemplates)

## Key Validation Functions

### StatefulSet
- `ValidateStatefulSet`: Full validation for create
- `ValidateStatefulSetUpdate`: Validates updates, allows changes only to replicas, ordinals, template, updateStrategy, revisionHistoryLimit, persistentVolumeClaimRetentionPolicy, minReadySeconds
- `ValidateStatefulSetSpec`: Validates pod management policy, update strategy, PVC retention policy, selector, template
- `ValidateStatefulSetStatus`: Validates status field constraints

### Deployment
- `ValidateDeployment`: Full validation for create
- `ValidateDeploymentUpdate`: Validates updates with immutable selector
- `ValidateDeploymentSpec`: Validates replicas, selector, template, strategy, progress deadline
- `ValidateDeploymentStrategy`: Validates Recreate or RollingUpdate configuration
- `ValidateRollingUpdateDeployment`: Ensures MaxUnavailable and MaxSurge are not both zero

### DaemonSet
- `ValidateDaemonSet`: Full validation for create
- `ValidateDaemonSetUpdate`: Validates updates with templateGeneration checks
- `ValidateDaemonSetSpec`: Validates selector, template, update strategy
- `ValidateRollingUpdateDaemonSet`: Ensures exactly one of MaxSurge/MaxUnavailable is non-zero

### ReplicaSet
- `ValidateReplicaSet`: Full validation for create
- `ValidateReplicaSetUpdate`: Validates updates with immutable selector
- `ValidateReplicaSetSpec`: Validates replicas, selector, template

### ControllerRevision
- `ValidateControllerRevisionCreate`: Validates revision with required data field
- `ValidateControllerRevisionUpdate`: Ensures data is immutable

## Helper Functions
- `ValidatePositiveIntOrPercent`: Validates int or percentage values
- `IsNotMoreThan100Percent`: Ensures percentage does not exceed 100%
- `ValidatePodTemplateSpecForStatefulSet`: Template validation with selector matching
- `ValidatePodTemplateSpecForReplicaSet`: Template validation for ReplicaSet/Deployment

## Common Validation Rules
- RestartPolicy must be Always for all workload controllers
- ActiveDeadlineSeconds is forbidden
- Selectors must be non-empty and match template labels
- Selectors are immutable after creation
