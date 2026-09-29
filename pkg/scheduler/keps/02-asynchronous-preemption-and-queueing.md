# Domain 2: Asynchronous Preemption & Scheduling Queuing

**Governing KEPs:**
- **KEP-4832**: *Asynchronous Preemption & Scheduler Async API Calls*
- **KEP-5142**: *Scheduling Queue Backoff and Unschedulable Queue Optimizations*

**Primary Packages:**
- `pkg/scheduler/backend/queue/`
- `pkg/scheduler/framework/preemption/`
- `pkg/scheduler/schedule_one.go`
- `pkg/scheduler/backend/api_dispatcher/`

---

## 1. Executive Summary & Problem Statement

In the legacy synchronous preemption model (KEP-562), when `DefaultPreemption` selected victim pods on a candidate node, the scheduler thread executed blocking HTTPS `DELETE` or `EVICT` calls directly to `kube-apiserver` within the scheduling cycle (`scheduleOne`). 

Under high-throughput cluster conditions or when evicting multiple victims across nodes, this synchronous design suffered from severe architectural deficiencies:
1. **Head-of-Line (HoL) Blocking**: The scheduler goroutine blocked on network round-trips to the API server and etcd serialization, degrading cluster-wide scheduling throughput from hundreds of pods/sec down to <10 pods/sec during heavy preemption storms.
2. **Victim Termination Latency**: While waiting for victim deletion acknowledgments, all other unscheduled pods in `activeQ` were delayed.
3. **Queue Starvation & Inefficient Requeuing**: Without intelligent pre-enqueue gating and backoff tracking (KEP-5142), unschedulable preemptor pods repeatedly cycled through `activeQ` -> `PostFilter` -> `unschedulablePods`, burning CPU and thrashing internal mutexes.

**KEP-4832** decoupled preemption evaluation from victim deletion actuation by introducing **Asynchronous Preemption**. The scheduling cycle identifies victims, sets nominations in memory, dispatches background deletion goroutines via an `AsyncPreemptionExecutor`, and immediately returns control to process the next pod in `activeQ`.

**KEP-5142** introduced fine-grained event-driven requeuing, exponential backoff structures, and pre-enqueue validation to ensure pods are only moved to `activeQ` when cluster events or victim terminations could plausibly make them schedulable.

---

## 2. Asynchronous Preemption Pipeline & Concurrency Architecture

```
                  +----------------------------------------------+
                  |           scheduleOne (Main Thread)          |
                  +----------------------------------------------+
                                         |
                                [PostFilter Evaluation]
                                         |
                          Identified Victims {V1, V2, V3}
                                         |
                       +-----------------+-----------------+
                       |                                   |
         (Synchronous Cache Update)              (Asynchronous Actuation)
                       |                                   |
           Set NominatedNodeName in Cache         Dispatch Background Goroutines
           Set PreEnqueue In-Flight Gate          (AsyncPreemptionExecutor Worker)
                       |                                   |
           Return Status: Unschedulable                    |
                       |                                   +---> Send DELETE V1
             Next Pod in activeQ!                          +---> Send DELETE V2
                                                           +---> Send DELETE V3
                                                                   |
                                                      +------------+------------+
                                                      |                         |
                                                 (All Succeeded)          (Any Failed != 404)
                                                      |                         |
                                            Victim Watchers Fire         Execute Rollback:
                                            Move Preemptor to activeQ    - Clear PreEnqueue Gate
                                                                         - Clear In-Memory Nomination
                                                                         - Re-queue to BackoffQ
```

### 2.1 AsyncPreemptionExecutor State Machine
The asynchronous actuation pipeline operates through the following stages:

1. **Pre-Eviction Gate Installation**:
   - The scheduler records the tuple `(PreemptorUID, CandidateNode, []VictimUIDs)` in the in-flight preemption registry.
   - Installs a `PreEnqueue` filter preventing the preemptor from re-entering `activeQ` until all victim deletions are dispatched or completed.
2. **Goroutine Worker Dispatch**:
   - A dedicated worker goroutine (bounded by a worker pool to avoid goroutine explosion) executes concurrent HTTP requests against `kube-apiserver`.
   - Each victim is deleted with its configured `GracePeriodSeconds`.
3. **Tolerating 404 Not Found & Concurrency Races**:
   - If a victim deletion returns `404 Not Found` (e.g., the victim pod finished running, was deleted by its controller, or was evicted by kubelet), the preemption worker treats this as a **success** rather than an error.
4. **Fast-Fail Rollback & Conflict Handling**:
   - If any victim deletion fails with an unrecoverable error (e.g., `403 Forbidden`, `500 Internal Server Error`, network timeout), the worker immediately halts further evictions for that tuple.
   - It performs an atomic rollback: clears the in-flight preemption state, removes the `nominatedNodeName` patch if not yet committed, and enqueues the preemptor into `backoffQ` with an incremented backoff penalty.

---

## 3. Scheduling Queue Backoff & Gating (KEP-5142)

The `SchedulingQueue` consists of three primary data structures:

1. **`activeQ`**: A priority heap ordered by `podInfo.Priority` (and arrival timestamp for ties).
2. **`backoffQ`**: A heap ordered by `podInfo.attempts` and `backoffUntil` timestamps. Pods that fail scheduling wait here until their backoff duration expires.
3. **`unschedulablePods`**: A key-value map (`pod.UID` -> `*framework.QueuedPodInfo`) storing pods that cannot make progress until relevant cluster events occur.

### 3.1 Event-Driven Moving from `unschedulablePods`
Instead of periodically flushing all unschedulable pods into `activeQ`, the queue registers specific cluster event triggers:
- **`PodDeleted` Event**: When a pod finishes terminating on a node, only unschedulable pods nominated for that node or requesting resources that the deleted pod occupied are moved to `activeQ` / `backoffQ`.
- **`NodeResourceFitChanged` Event**: When node allocatable capacity increases or taints are removed.
- **`AssignedPodUpdated` Event**: When running pods resize or release claims.

### 3.2 In-Flight Preemption Queue Invariant
To prevent thundering herd behavior when a high-priority preemptor is waiting for multiple victims:
- While async deletions are in flight, the preemptor is held in `unschedulablePods` guarded by `isPreemptionInFlight(pod) == true`.
- Once all victim deletion responses are acknowledged, the completion handler triggers `queue.MoveAllToActiveOrBackoffQueue(QueueSortPlugin, AssignedPodDelete, ...)`.

---

## 4. Concurrency Invariants & Synchronization Primitives

1. **In-Memory WaitingPod Cancellation**:
   - If a victim is currently in the `Permit` phase (a `WaitingPod` waiting for co-scheduling gang members), preemption cancels the waiting permit in-memory via `waitingPod.Reject("Preempted")`. No network round-trip is issued.
2. **Atomic Nomination Verification**:
   - Before binding a nominated pod, the scheduler must verify that the node snapshot still has sufficient capacity (i.e., that other lower-priority pods did not steal the capacity while victims were terminating).
3. **Graceful Deletion Collision Prevention**:
   - If two independent preemptors ($P_1$ and $P_2$) run `PostFilter` concurrently across different profiles, the scheduler cache uses mutex synchronization on node state objects to ensure the same victim cannot be allocated to both preemptors.

---

## 5. Failure Modes & Edge Cases

1. **Orphaned Nominations**: If the scheduler crashes while async preemption goroutines are in flight, the new scheduler leader clears stale `nominatedNodeName` fields on pods whose victims no longer exist.
2. **Stuck Terminating Victims**: If a victim pod has a non-zero termination grace period and finalizers that block indefinitely, the preemptor will remain unscheduled until finalizers are cleared.
3. **Preemption Thundering Herd**: Multiple high-priority pods vying for the same node. Bounded worker pools and backoff multipliers prevent API server saturation.

---

## 6. Verification & Test Matrix

- **Unit Tests**: `pkg/scheduler/backend/queue/scheduling_queue_test.go`
- **Integration Tests**: `test/integration/scheduler/preemption/`
  - `TestAsyncPreemptionActuation`: Validates goroutine decoupling and non-blocking scheduling of subsequent pods.
  - `TestAsyncPreemptionRollbackOnDeletionFailure`: Ensures atomic cleanup when deletion fails.
  - `TestAsyncPreemptionTolerate404NotFound`: Validates idempotency when victims are deleted concurrently.
