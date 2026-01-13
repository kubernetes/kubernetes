# Package: ttlafterfinished

## Purpose
The TTL After Finished controller automatically cleans up finished Jobs after their TTL has expired. It watches Jobs with a non-nil `.spec.ttlSecondsAfterFinished` and deletes them once the specified time has passed since completion.

## Key Types

- **Controller**: Main controller that watches Jobs, tracks TTL expiration, and deletes expired Jobs.

## Key Functions

- **New(ctx, jobInformer, client)**: Creates a new TTL After Finished controller instance.
- **Run(ctx, workers)**: Starts the controller workers.
- **addJob/updateJob**: Event handlers that enqueue Jobs needing cleanup.
- **processJob(ctx, key)**: Checks if a Job's TTL has expired and deletes it if so.
- **processTTL(logger, job)**: Calculates remaining TTL and either returns expiration time or re-enqueues the Job.
- **needsCleanup(job)**: Checks if a Job is finished and has TTL configured.
- **jobFinishTime(job)**: Extracts the finish time from Job conditions.
- **timeLeft(logger, job, since)**: Calculates time remaining until TTL expiration.

## Design Notes

- Jobs are deleted using foreground cascade deletion (deletes pods first).
- Uses preconditions with UID to prevent deleting recreated Jobs.
- Re-enqueues Jobs with `AddAfter` to check again when TTL is expected to expire.
- Performs a fresh API lookup before deletion to handle stale cache data.
- Records deletion latency metrics via the metrics subpackage.
