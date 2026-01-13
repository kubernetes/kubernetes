# Package: job

## Purpose
Implements the registry strategy for Job resources, which run pods to completion and track success/failure counts.

## Key Types

- **jobStrategy**: Implements verification logic and REST strategies for Jobs
- **jobStatusStrategy**: Strategy for status-only updates with extensive validation for external controllers

## Key Functions

- **Strategy**: Default logic for creating/updating Job objects
- **StatusStrategy**: Default logic for status updates with strengthened validation
- **PrepareForCreate()**: Generates selector/labels, clears status, drops disabled fields (ManagedBy, SuccessPolicy, BackoffLimitPerIndex, PodReplacementPolicy)
- **PrepareForUpdate()**: Preserves status, increments Generation on spec changes
- **Validate()**: Validates new Jobs with extensive options
- **generateSelector()**: Auto-generates pod selector and labels from job UID
- **DefaultGarbageCollectionPolicy()**: Returns OrphanDependents for batch/v1 (backwards compat)
- **JobToSelectableFields()**: Returns fields.Set including status.successful for filtering
- **MatchJob()**: Creates SelectionPredicate for efficient etcd watch routing
- **getStatusValidationOptions()**: Extensive validation when JobManagedBy feature is enabled

## Design Notes

- Namespace-scoped resource
- Supports many feature gates: JobManagedBy, JobSuccessPolicy, JobBackoffLimitPerIndex, JobPodReplacementPolicy, MutablePodResourcesForSuspendedJobs
- Selector is auto-generated using controller-uid label unless ManualSelector=true
- Status validation is significantly stricter when JobManagedBy is enabled (external controllers)
- Supports field selectors on status.successful
