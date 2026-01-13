# Package: tainteviction

Implements the TaintEviction controller (NoExecute taint manager) that evicts pods from nodes with NoExecute taints.

## Key Types

- **Controller**: Listens to taint/toleration changes and schedules pod deletions.
- **TimedWorkerQueue**: Manages delayed pod deletion with cancellation support.
- **nodeUpdateItem/podUpdateItem**: Work items for node and pod updates.

## Key Functions

- **New**: Creates a new controller with pod and node informers.
- **Run**: Starts workers processing node and pod updates.
- **NodeUpdated**: Queues node for processing when taints change.
- **PodUpdated**: Queues pod for processing when tolerations change or pod is scheduled.
- **handleNodeUpdate**: Processes all pods on a node when its taints change.
- **handlePodUpdate**: Evaluates if pod should be evicted based on current taints.
- **processPodOnNode**: Core logic matching taints to tolerations and scheduling eviction.

## Key Constants

- **NodeUpdateChannelSize**: 10 (buffer for node updates)
- **UpdateWorkerSize**: 8 workers for processing updates
- **retries**: 5 attempts for pod deletion

## Design Patterns

- Uses hash-based worker sharding by node name for consistency.
- Prioritizes node updates over pod updates to handle taints quickly.
- Supports toleration-based eviction delays (TolerationSeconds).
- Cancels scheduled evictions when tolerations change or taints are removed.
- Marks pods with DisruptionTarget condition before deletion.
- Exposes metrics for pod deletions and latency.
