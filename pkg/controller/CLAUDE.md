# Package: controller

## Purpose
Provides shared utilities and base functionality for all Kubernetes controllers, including expectations tracking, pod/node management, and controller lifecycle helpers.

## Key Types/Structs
- `ControllerExpectations`: TTL cache tracking expected pod creates/deletes per controller
- `ControlleeExpectations`: Atomic counters for tracking individual controller's expectations
- `PodControlInterface`: Interface for creating/deleting pods with proper ownership
- `ControllerRevisionControlInterface`: Interface for managing controller revisions

## Key Functions
- `NewControllerExpectations()`: Creates new expectations tracker with TTL
- `SatisfiedExpectations(key)`: Returns true when expected creates/deletes are observed or expired
- `KeyFunc`: DeletionHandlingMetaNamespaceKeyFunc for generating cache keys
- `NoResyncPeriodFunc` / `StaticResyncPeriodFunc`: Helpers for resync period configuration

## Key Constants
- `ExpectationsTimeout`: 5 minutes before expectations expire
- `SlowStartInitialBatchSize`: Initial batch size (1) for slow-start pod creation
- `PodNodeNameKeyIndex`: Index name for pod-by-node lookups
- `PodControllerIndex`: Index name for pod-by-controller lookups

## Design Notes
- Expectations prevent duplicate creates/deletes when informer cache is stale
- Controllers should not sync until expectations are satisfied or expired
- Provides common patterns: workqueue, informers, rate limiting, pod management
- Used by DaemonSet, ReplicaSet, StatefulSet, Job, and other controllers
