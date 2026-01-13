# Package: batch/v1

## Purpose
Provides defaulting functions and scheme registration for the batch/v1 API version, which is the stable version for Jobs and CronJobs.

## Key Defaulting Functions

### SetDefaults_Job
- **Completions**: Defaults to 1 if parallelism is also unset
- **Parallelism**: Defaults to 1
- **BackoffLimit**: Defaults to 6 (with backoffLimitPerIndex feature: defaults to MaxInt32 if perIndex limit is set)
- **CompletionMode**: Defaults to NonIndexedCompletion
- **Suspend**: Defaults to false
- **PodReplacementPolicy**: Defaults to TerminatingOrFailed (or Failed when PodFailurePolicy is set)
- **ManagedBy**: Defaults to `kubernetes.io/job-controller`

### SetDefaults_CronJob
- **ConcurrencyPolicy**: Defaults to AllowConcurrent
- **Suspend**: Defaults to false
- **SuccessfulJobsHistoryLimit**: Defaults to 3
- **FailedJobsHistoryLimit**: Defaults to 1

## Key Functions
- **addDefaultingFuncs()**: Registers defaulting functions with the scheme

## Design Notes
- Defaults interact with feature gates (BackoffLimitPerIndex)
- PodReplacementPolicy default changes based on whether PodFailurePolicy is configured
- Completions and Parallelism have interdependent defaulting logic
