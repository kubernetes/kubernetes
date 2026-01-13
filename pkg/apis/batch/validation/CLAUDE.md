# Package: batch/validation

## Purpose
Provides comprehensive validation for Job and CronJob resources, ensuring configurations are valid before being persisted.

## Key Validation Functions

### Job Validation
- **ValidateJob()**: Full validation of a Job object
- **ValidateJobSpec()**: Validates JobSpec fields including parallelism, completions, selectors
- **ValidateJobUpdate()**: Validates updates with immutability checks
- **ValidateJobStatusUpdate()**: Validates status-only updates

### CronJob Validation
- **ValidateCronJob()**: Full validation of a CronJob object
- **ValidateCronJobSpec()**: Validates schedule, concurrency policy, history limits
- **ValidateCronJobUpdate()**: Validates CronJob updates

### Specialized Validators
- **validatePodFailurePolicy()**: Validates pod failure policy rules (exit codes, conditions)
- **validateSuccessPolicy()**: Validates success policy rules (indexes, counts)
- **validateCompletionMode()**: Validates Indexed vs NonIndexed modes
- **validateJobTemplateSpec()**: Validates embedded pod template

## Key Validation Rules
- Parallelism and completions must be non-negative
- BackoffLimit must be non-negative (max 10^8)
- Indexed jobs require completions and have max 10^5 completions
- Pod failure policy actions must be valid (FailJob, Ignore, Count, FailIndex)
- Exit codes in failure policy must be 0-255 and unique
- Schedule must be valid cron format (standard or TZ-prefixed)
- Certain fields are immutable after creation (completions for indexed jobs, completionMode, selector)

## Design Notes
- Uses field.ErrorList for detailed error reporting with field paths
- Supports both create and update validation with different rules
- Validates feature-gated fields conditionally
- Extensive validation for indexed job completion indexes format
