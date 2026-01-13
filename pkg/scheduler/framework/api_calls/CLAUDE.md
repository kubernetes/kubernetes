# Package: apicalls

## Purpose
Defines the API call types used by the scheduler for pod binding and status updates. Provides implementations of the framework.APICall interface for different types of Kubernetes API operations.

## Key Types

### PodBindingCall
Represents a pod binding operation that assigns a pod to a node:
- **binding**: The v1.Binding object containing pod and target node information
- Implements Execute() to call the Kubernetes Bind API
- Returns the binding UID for tracking

### PodStatusPatchCall
Represents a pod status patch operation:
- **podStatus**: Current pod status
- **newCondition**: Condition to add/update
- **nominatingInfo**: Node nomination information for preemption
- Supports merging multiple patches to the same pod
- Tracks execution state to handle concurrent sync operations

## Key Constants

- **PodStatusPatch**: API call type identifier for status patches (relevance: 1)
- **PodBinding**: API call type identifier for bindings (relevance: 2)

## Key Variables

- **Relevances**: Maps API call types to their priority/relevance (higher means more important)
- **Implementations**: Maps types to constructor functions for creating call objects

## Key Functions

- **NewPodBindingCall(binding)**: Creates a new binding call
- **NewPodStatusPatchCall(pod, condition, nominatingInfo)**: Creates a new status patch call
- **Execute(ctx, client)**: Executes the API call
- **Sync(obj)**: Updates the call with latest object state
- **Merge(oldCall)**: Combines with a previous call to the same object
- **IsNoOp()**: Returns true if the call would have no effect

## Design Pattern
- Implements the command pattern for deferred API execution
- Supports call merging for batching multiple updates
- Thread-safe with internal mutex for concurrent access
