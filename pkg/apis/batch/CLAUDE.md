# Package: batch

## Purpose
Defines internal (unversioned) types for the Kubernetes batch API group, which manages Jobs and CronJobs for running batch workloads.

## Key Types
- **Job**: Represents a batch job that runs pods to completion
- **JobSpec**: Configuration including parallelism, completions, backoffLimit, completionMode, podFailurePolicy, successPolicy
- **CronJob**: Schedules Jobs on a time-based schedule (cron format)
- **CronJobSpec**: Schedule configuration with concurrency policy and history limits
- **PodFailurePolicy**: Rules for handling pod failures (exit codes, conditions)
- **SuccessPolicy**: Rules for early Job success based on succeeded indexes
- **JobCondition**: Status condition types (Complete, Failed, Suspended, FailureTarget, SuccessCriteriaMet)

## Key Features
- **Completion Modes**: NonIndexed (default) and Indexed (for parallel indexed jobs)
- **Pod Failure Policy**: Fine-grained control over failure handling with actions (FailJob, Ignore, Count, FailIndex)
- **Success Policy**: Allow jobs to succeed early when specific indexes complete
- **Pod Replacement Policy**: Controls when replacement pods are created (TerminatingOrFailed, Failed)
- **Managed-by field**: Allows external controllers to manage Jobs

## Key Functions
- `IsJobFinished()`: Checks if job has Complete or Failed condition
- `IsKubeletPodmgmtManagedJob()`: Checks if job is managed by the kubelet via PodCertificateRequest
- `JobConditionType` constants for various job states

## Design Notes
- Internal types are converted to/from versioned types (v1, v1beta1) for API compatibility
- Supports both simple single-pod jobs and complex parallel indexed jobs
