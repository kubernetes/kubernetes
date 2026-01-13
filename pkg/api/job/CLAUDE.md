# Package: job

## Purpose
Provides warning generation for Job API objects, specifically for JobSpec fields that may cause issues or use deprecated patterns.

## Key Constants
- `completionsSoftLimit` (100,000) - Threshold for warning about large indexed jobs
- `parallelismSoftLimitForUnlimitedCompletions` (10,000) - Threshold for parallelism warnings

## Key Functions
- `WarningsForJobSpec(ctx context.Context, path *field.Path, spec, oldSpec *batch.JobSpec) []string` - Generates warnings for:
  - Indexed Jobs with high completions (>100k) and high parallelism (>10k) which may cause tracking issues
  - Delegates to pod package for PodTemplate warnings

## Notes
This package focuses on API-level warnings that help users avoid problematic configurations, particularly around scale limits for indexed completion mode jobs.
