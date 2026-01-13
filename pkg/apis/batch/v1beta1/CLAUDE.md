# Package: batch/v1beta1

## Purpose
Provides defaulting functions for the deprecated batch/v1beta1 API version, which only includes CronJob (Jobs are not in v1beta1).

## Key Defaulting Functions

### SetDefaults_CronJob
- **ConcurrencyPolicy**: Defaults to AllowConcurrent
- **Suspend**: Defaults to false
- **SuccessfulJobsHistoryLimit**: Defaults to 3
- **FailedJobsHistoryLimit**: Defaults to 1

## Design Notes
- v1beta1 only contains CronJob; Job was promoted directly to v1
- This version is deprecated; use batch/v1 instead
- Defaulting logic mirrors batch/v1 CronJob defaults
- Minimal implementation as most batch features are in v1
