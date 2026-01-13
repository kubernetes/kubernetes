# Package: util

Utility functions for the Deployment controller.

## Key Constants

- `RevisionAnnotation`: Annotation key for tracking rollout revision sequence
- `DesiredReplicasAnnotation`: Records desired replicas on ReplicaSets
- `MaxReplicasAnnotation`: Maximum replicas during surge (spec.replicas + maxSurge)

## Key Functions

- `GetNewReplicaSet()`: Finds the new ReplicaSet for a Deployment
- `GetOldReplicaSets()`: Finds old ReplicaSets for a Deployment
- `SetNewReplicaSetAnnotations()`: Copies annotations from Deployment to ReplicaSet
- `MaxRevision()`: Returns the highest revision number among ReplicaSets
- `MaxSurge()`: Calculates maximum surge pods allowed
- `MaxUnavailable()`: Calculates maximum unavailable pods allowed
- `DeploymentComplete()`: Checks if a Deployment has completed rollout
- `DeploymentProgressing()`: Checks if a Deployment is making progress
- `DeploymentTimedOut()`: Checks if a Deployment has exceeded its progress deadline

## Purpose

Provides helper functions for managing Deployments, ReplicaSets, and their relationships. Handles revision tracking, rollout calculations, status conditions, and replica management.

## Design Notes

- Extensive use of annotations for tracking deployment metadata
- Supports both absolute and percentage-based surge/unavailable values
- Progress tracking based on replica counts and conditions
