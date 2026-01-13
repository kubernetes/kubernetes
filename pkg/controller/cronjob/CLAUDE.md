# Package: cronjob

## Purpose
Implements the CronJob controller (v2) that creates Jobs on a time-based schedule using informers and DelayingQueue.

## Key Types/Structs
- `ControllerV2`: Main controller with workqueue, Job/CronJob listers, and control interfaces
- `jobControlInterface`: Interface for creating/deleting Jobs
- `cjControlInterface`: Interface for updating CronJob status

## Key Functions
- `NewControllerV2(ctx, jobInformer, cronJobsInformer, kubeClient)`: Creates the controller
- `Run(ctx, workers)`: Starts the controller with specified worker count
- `sync(ctx, cronJobKey)`: Main sync loop for a single CronJob
- `syncCronJob(ctx, cronJob, jobs)`: Processes schedule, creates/cleans up Jobs

## Scheduling Logic
1. Calculate next scheduled time based on cron expression
2. Check concurrency policy (Allow, Forbid, Replace)
3. Handle missed schedules within StartingDeadlineSeconds
4. Create Jobs for scheduled times
5. Clean up old Jobs based on SuccessfulJobsHistoryLimit/FailedJobsHistoryLimit

## Concurrency Policies
- `Allow`: Multiple Jobs can run concurrently
- `Forbid`: Skip new Job if previous is still running
- `Replace`: Delete running Job and create new one

## Design Notes
- Uses DelayingQueue to schedule wake-ups at precise times
- 100ms delta for next schedule calculations
- Watches both Jobs and CronJobs for changes
- Records events for missed schedules and Job lifecycle
