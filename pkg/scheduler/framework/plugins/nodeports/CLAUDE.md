# Package: nodeports

## Purpose
Implements a filter plugin that checks if a node has free host ports for all the ports requested by a pod's containers.

## Key Types

### NodePorts
The plugin struct implementing:
- PreFilterPlugin, FilterPlugin, EnqueueExtensions, SignPlugin
- **enableSchedulingQueueHint**: Feature flag for queueing hints

### preFilterState
Cached state containing the list of host ports requested by the pod.

## Extension Points

### PreFilter
- Extracts all host ports from pod's containers
- If no host ports requested, returns Skip status
- Caches port list in CycleState for Filter phase

### Filter
- Checks each requested host port against ports already used on the node
- Returns Unschedulable if any port conflicts exist
- Checks both port number and protocol (TCP/UDP/SCTP)

## Key Functions

- **New(ctx, obj, handle, features)**: Creates the plugin
- **SignPod(ctx, pod)**: Returns host ports for pod signing
- **EventsToRegister()**: Returns Node/Add and Pod/Delete events
- **getContainerPorts(pod)**: Extracts all host ports from containers

## Port Conflict Detection
Checks for conflicts based on:
- Host port number
- Protocol (TCP, UDP, SCTP)
- Host IP (0.0.0.0 conflicts with any IP)

## Design Pattern
- Uses precomputed state to avoid repeated port extraction
- Returns Unschedulable (not UnschedulableAndUnresolvable) because pod deletions can free ports
- Considers all container types: init containers, regular containers, ephemeral containers
