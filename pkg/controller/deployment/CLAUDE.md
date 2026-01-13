# Package: deployment

Kubernetes Deployment controller implementation.

## Key Types

- `DeploymentController`: Main controller struct that synchronizes Deployment objects with ReplicaSets and Pods

## Key Functions

- `NewDeploymentController()`: Creates and initializes a new DeploymentController
- `Run()`: Starts the controller's reconciliation loop
- `syncDeployment()`: Main reconciliation logic for a single Deployment

## Purpose

Manages Kubernetes Deployments by creating and scaling ReplicaSets to match the desired state. Implements deployment strategies (rolling update, recreate), rollback capabilities, proportional scaling for risk mitigation, and cleanup policies.

## Key Features

- Rolling update strategy with configurable max surge and max unavailable
- Recreate strategy for complete replacement
- Automatic rollback on failures
- Revision history management
- Progress deadline tracking

## Design Notes

- Uses informers for Deployments, ReplicaSets, and Pods
- Workqueue-based reconciliation with rate limiting (max 15 retries)
- Adopts/releases ReplicaSets based on controller references
- Implements the standard Kubernetes controller pattern
