# Package: cronjob

## Purpose
Implements the registry strategy for CronJob resources, which run Jobs on a time-based schedule (like Unix cron).

## Key Types

- **cronJobStrategy**: Implements verification logic and REST strategies for CronJobs
- **cronJobStatusStrategy**: Strategy for status-only updates

## Key Functions

- **Strategy**: Default logic for creating/updating CronJob objects
- **StatusStrategy**: Default logic for status updates
- **PrepareForCreate()**: Clears status, sets Generation=1, drops disabled template fields
- **PrepareForUpdate()**: Preserves status, increments Generation on spec changes
- **Validate()**: Validates new CronJobs with pod template validation
- **ValidateUpdate()**: Validates CronJob updates
- **WarningsOnCreate/Update()**: Warns about DNS label issues and deprecated TZ in schedule
- **DefaultGarbageCollectionPolicy()**: Returns OrphanDependents for batch/v1beta1 (backwards compat), DeleteDependents for others

## Design Notes

- Namespace-scoped resource
- Uses job.WarningsForJobSpec to warn about embedded job spec issues
- Warns if schedule uses TZ/CRON_TZ (should use timeZone field instead)
- Status updates reset spec to prevent modification
- Supports both batch/v1 and batch/v1beta1 API versions
