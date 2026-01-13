# Package: gangscheduling

## Purpose
Implements gang scheduling for Kubernetes, enabling "all-or-nothing" scheduling of pod groups. Ensures that all pods in a gang are scheduled together or none are scheduled.

## Key Types

### GangScheduling
The plugin struct implementing:
- **EnqueueExtensions**: Registers events for queue management
- **PreEnqueuePlugin**: Gates pods until gang quorum is met
- **ReservePlugin**: Tracks assumed pods in the gang
- **PermitPlugin**: Holds pods until gang is complete

## Extension Points

### PreEnqueue
- Checks if pod's Workload object exists
- Verifies the pod group has a Gang scheduling policy
- Blocks scheduling until MinCount pods from the gang are available
- Returns UnschedulableAndUnresolvable if quorum not met

### Reserve
- Marks the pod as "assumed" in WorkloadManager
- Contributes to the count of pods ready for co-scheduling

### Unreserve
- Removes pod from assumed state if scheduling fails
- Called when any subsequent scheduling step fails

### Permit
- Blocks until MinCount pods from the gang are assumed or assigned
- Returns Wait status with timeout if quorum not yet met
- When quorum is met, allows all waiting pods from the gang
- Activates unscheduled gang pods to expedite gang formation

## Key Functions

- **New(ctx, obj, handle, features)**: Creates the plugin
- **EventsToRegister()**: Returns Pod/Add and Workload/Add events
- **isSchedulableAfterPodAdded**: Queueing hint for new gang pods
- **isSchedulableAfterWorkloadAdded**: Queueing hint for new workloads

## Design Pattern
- Uses WorkloadManager to track pod group state across scheduling cycles
- 5-minute default timeout for gang formation
- Relies on Workload API (scheduling/v1alpha1) for gang configuration
- Gang pods wait at Permit stage until quorum is achieved
