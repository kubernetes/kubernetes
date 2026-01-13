# Package: queue

## Purpose
Implements the scheduler's priority queue system for managing pods awaiting scheduling. This is a core component that determines which pods get scheduled and in what order.

## Key Types

### PriorityQueue
The main scheduling queue implementation with three sub-queues:
- **activeQ**: Heap of pods ready to be scheduled (highest priority first)
- **backoffQ**: Pods in exponential backoff after failed scheduling attempts
- **unschedulablePods**: Pods that couldn't be scheduled and are waiting for cluster events

### Nominator
Tracks nominated nodes for pods during preemption. Stores which pods have been nominated to run on which nodes.

### UnschedulablePods
Stores pods that failed scheduling with their failure reasons. Tracks which plugins caused the failure for event-based requeueing.

## Key Functions

- **NewPriorityQueue(lessFn, informerFactory, opts...)**: Creates a new queue
- **Add(pod)**: Adds a pod to the active queue
- **Pop(logger)**: Removes and returns the highest priority pod (blocks if empty)
- **AddUnschedulableIfNotPresent(pInfo, gated)**: Moves failed pods to appropriate queue
- **MoveAllToActiveOrBackoffQueue(event, oldObj, newObj, preCheck)**: Moves pods when cluster events occur
- **Activate(logger, pods)**: Forces specific pods into active queue

## Design Pattern
- Event-driven requeueing: pods move between queues based on cluster events (node added, pod deleted, etc.)
- Exponential backoff for repeatedly failing pods
- QueueingHint functions allow plugins to specify which events can make a pod schedulable
- Per-profile queues when multiple scheduler profiles are configured

## Important Notes
- Uses heap-based priority queue with customizable Less function
- Supports pod nomination tracking for preemption
- Thread-safe with internal mutex protection
- Integrates with metrics for monitoring queue health
