# Package: framework

## Purpose
Defines the core scheduling framework interfaces, types, and utilities. This package contains the extension points and data structures that plugins implement and the scheduler uses to make scheduling decisions.

## Key Types

### Framework (interface)
The main interface for the scheduling framework. Extends `fwk.Handle` with methods for:
- Running plugin extension points (PreFilter, Filter, PostFilter, Score, Reserve, Permit, PreBind, Bind, PostBind)
- Queue management (QueueSortFunc, PreEnqueuePlugins)
- Pod signing for result caching (SignPod, GetNodeHint, StoreScheduleResults)
- Plugin introspection (HasFilterPlugins, HasScorePlugins, ListPlugins)

### NodeToStatus
Maps node names to scheduling statuses. Tracks why pods can't be scheduled on specific nodes with support for "absent nodes" (default status for unlisted nodes).

### CycleState
Thread-safe key-value store for sharing data between plugins during a scheduling cycle. Uses sync.Map for concurrent access with support for:
- Read/Write/Delete operations
- Cloning for parallel operations
- Skip plugin tracking

### PodsToActivate
State data for tracking pods that should be moved to activeQ. Used by plugins to trigger requeueing of related pods.

## Key Functions

- **NewCycleState()**: Creates a new cycle state
- **NewDefaultNodeToStatus()**: Creates NodeToStatus with default "UnschedulableAndUnresolvable" for absent nodes
- **PodSchedulingPropertiesChange(newPod, oldPod)**: Interprets pod updates and returns relevant cluster events
- **NodeSchedulingPropertiesChange(newNode, oldNode)**: Interprets node updates and returns relevant cluster events

## Event Constants
- **ScheduleAttemptFailure**: When scheduling fails
- **BackoffComplete**: When a pod finishes backoff
- **ForceActivate**: When a pod is forcibly moved to activeQ
- **UnschedulableTimeout**: When a pod times out in unschedulable queue

## Design Pattern
- Plugin-based architecture with well-defined extension points
- Event-driven requeueing based on cluster changes
- State sharing via CycleState with cloneable data
