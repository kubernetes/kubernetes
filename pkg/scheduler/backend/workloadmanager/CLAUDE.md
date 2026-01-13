# Package: workloadmanager

## Purpose
Manages the state of pods belonging to Workload objects for gang scheduling. Tracks pod groups and their scheduling states to enable "all-or-nothing" scheduling of related pods.

## Key Types

### WorkloadManager (interface)
Extends `framework.WorkloadManager` with methods for handling pod lifecycle events:
- **AddPod(pod)**: Called when a Pod/Add event is observed
- **UpdatePod(oldPod, newPod)**: Called when a Pod/Update event is observed
- **DeletePod(pod)**: Called when a Pod/Delete event is observed

### workloadManager (struct)
Concrete implementation storing:
- **podGroupInfos**: Map of podGroupKey to podGroupInfo for each known pod group

### podGroupKey
Identifies a specific pod group instance by namespace, workload name, pod group name, and replica key.

### podGroupInfo
Runtime state of a pod group tracking:
- **allPods**: All pods belonging to the group
- **unscheduledPods**: Pods not yet scheduled (neither assumed nor assigned)
- **assumedPods**: Pods that passed Reserve and are waiting at Permit
- **assignedPods**: Pods that are bound to nodes
- **schedulingDeadline**: Timeout for gang formation

## Key Functions

- **New()**: Creates a new workload manager
- **PodGroupInfo(namespace, workloadRef)**: Returns state for a pod group
- **AssumePod(podUID)**: Marks a pod as having reached Reserve stage
- **ForgetPod(podUID)**: Removes pod from assumed state
- **SchedulingTimeout()**: Returns remaining time until gang scheduling times out

## Design Pattern
- Driven by scheduler event handlers for thread safety
- Supports gang scheduling by tracking assumed/assigned pod counts
- Uses a 5-minute default timeout for gang formation
- Assumes pod.Spec.WorkloadRef is immutable
