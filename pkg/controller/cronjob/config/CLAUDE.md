# Package: cronjob/config

## Purpose
Defines configuration types for the CronJob controller.

## Key Types/Structs
- `CronJobControllerConfiguration`: Configuration struct with ConcurrentCronJobSyncs field

## Configuration Fields
- `ConcurrentCronJobSyncs`: Number of CronJob objects that can be synced concurrently
  - Higher values = more responsive job creation
  - Higher values = more CPU and network load
  - Controls worker concurrency in the controller

## Design Notes
- Simple configuration with single tuning parameter
- Part of the controller manager's overall configuration
- Trade-off between responsiveness and resource usage
