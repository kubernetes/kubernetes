# Package: util

Utility functions for the Job controller.

## Key Functions

- `FinishedCondition()`: Returns whether a job is finished and its condition type (Complete or Failed)
- `IsJobFinished()`: Checks if a job has finished execution (success or failure)
- `IsJobSucceeded()`: Returns true only if job completed successfully

## Purpose

Provides helper functions for checking Job status. These utilities are used throughout the Job controller and related components to determine job completion state.

## Design Notes

- Finished state is determined by checking for JobComplete or JobFailed conditions with status True
- Simple utility package with no dependencies on controller internals
