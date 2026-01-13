# Package: disruption

PodDisruptionBudget (PDB) controller implementation.

## Key Types

- `DisruptionController`: Manages PodDisruptionBudget objects, tracking disruption counts and allowed disruptions

## Key Constants

- `DeletionTimeout`: 2 minutes - maximum time a pod can be in DisruptedPods before being considered not deleted
- `stalePodDisruptionTimeout`: 2 minutes - maximum time for stale DisruptionTarget condition

## Key Functions

- `NewDisruptionController()`: Creates the controller with informers for PDBs, Pods, ReplicaSets, Deployments, etc.
- `Run()`: Starts the reconciliation loop
- `sync()`: Main reconciliation logic for a PDB

## Purpose

Ensures that voluntary disruptions (like node drains, pod deletions) respect PodDisruptionBudget constraints. The controller continuously calculates and updates the `status.disruptionsAllowed` field based on the number of healthy pods matching the PDB selector.

## Key Features

- Tracks disrupted pods and their deletion status
- Calculates allowed disruptions based on minAvailable/maxUnavailable
- Handles unhealthy pod eviction policy
- Cleans up stale disruption conditions

## Design Notes

- Uses scale subresource to get replica counts for workloads
- Requires clock synchronization between nodes for accurate timeout handling
- Integrates with various workload types: Deployments, ReplicaSets, StatefulSets, ReplicationControllers
