# Package: job

Kubernetes Job controller implementation.

## Key Types

- `Controller`: Main controller ensuring Jobs have corresponding pods to run workloads
- `syncJobCtx`: Context for a single Job sync operation

## Key Constants

- `MaxUncountedPods`: 500 - max pods in uncountedTerminatedPods
- `MaxPodCreateDeletePerSync`: 500 - max pod operations per sync

## Key Functions

- `NewController()`: Creates the Job controller with pod and job informers
- `Run()`: Starts the reconciliation loop
- `syncJob()`: Main reconciliation logic for a single Job

## Purpose

Manages Kubernetes Jobs by creating pods to completion and tracking job success/failure status. Supports indexed jobs, pod failure policies, success policies, and backoff for failed pods.

## Key Features

- Indexed completion mode for parallel jobs with specific indices
- Pod failure policy for custom failure handling (FailJob, Ignore, Count, FailIndex)
- Success policy for early job completion
- Exponential backoff for pod recreation after failures
- BackoffLimitPerIndex for per-index failure limits
- Pod tracking via finalizers

## Design Notes

- Uses workqueue with exponential backoff
- Maintains expectations for pod creates/deletes
- Tracks terminated pods via finalizers for accurate counting
- Supports both NonIndexed and Indexed completion modes
