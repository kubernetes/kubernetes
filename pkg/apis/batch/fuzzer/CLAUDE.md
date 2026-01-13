# Package: batch/fuzzer

## Purpose
Provides fuzz testing functions for the batch API types to ensure proper serialization roundtrip behavior during API testing.

## Key Functions
- **Funcs()**: Returns fuzzer functions for the batch API group

## Fuzzer Functions
1. **JobSpec fuzzer**: Sets valid defaults for job specs:
   - Completions: 1
   - Parallelism: 1
   - BackoffLimit: 6
   - CompletionMode: NonIndexedCompletion
   - PodReplacementPolicy: TerminatingOrFailed
   - ManualSelector: false

2. **CronJobSpec fuzzer**: Sets valid cron job defaults:
   - Schedule: "*/5 * * * ?"
   - ConcurrencyPolicy: AllowConcurrent
   - Suspend: false
   - SuccessfulJobsHistoryLimit: 3
   - FailedJobsHistoryLimit: 1

3. **PodFailurePolicyRule fuzzer**: Sets action to FailJob and generates valid OnExitCodes or OnPodConditions

4. **SuccessPolicyRule fuzzer**: Generates valid SucceededIndexes strings and SucceededCount values

## Design Notes
- Fuzzer ensures fields have valid values that pass validation
- Used by `pkg/api/testing/serialization_test.go` for roundtrip tests
- Prevents nil pointer issues and invalid enum values during fuzz testing
