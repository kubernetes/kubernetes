# PrioritySort Plugin (`pkg/scheduler/framework/plugins/queuesort`)

This guide provides an architectural overview, interface implementations, queue comparison semantics, ordering invariants, and testing strategies for the `PrioritySort` plugin in `pkg/scheduler/framework/plugins/queuesort`.

---

## 1. High-Level Purpose & Scope

The `PrioritySort` plugin provides the default ordering logic for the Kubernetes scheduler's active scheduling queue (`activeQ`). It establishes the comparator function used by the underlying priority heap to determine which queued entity (Pod or PodGroup) is popped next for scheduling.

### Core Responsibilities:
1. **Priority-Based Precedence**: Ensures higher-priority workloads are dequeued and scheduled before lower-priority workloads.
2. **Deterministic FIFO Tie-Breaking**: For entities with identical priority levels, breaks ties using the timestamp at which the entity entered the scheduling queue, ensuring fair First-In, First-Out (FIFO) ordering.
3. **Queued Entity Abstraction**: Operates on `fwk.QueuedEntityInfo`, supporting both individual pods (`QueuedPodInfo`) and generic workloads / pod groups.

---

## 2. Package Architecture & File Map

```
pkg/scheduler/framework/plugins/queuesort/
├── priority_sort.go       # Plugin definition, Less implementation, New factory
├── priority_sort_test.go  # Unit tests for priority comparison and timestamp ordering
└── AGENTS.md              # This agent documentation
```

---

## 3. Extension Point Implementations

`PrioritySort` implements the `fwk.QueueSortPlugin` interface.

```
                  ┌───────────────────────────────────────────────┐
                  │          activeQ Heap Pop / Push              │
                  └───────────────────────┬───────────────────────┘
                                          │
                                          ▼
                  ┌───────────────────────────────────────────────┐
                  │             PrioritySort.Less                 │
                  │         (entity1, entity2)                    │
                  └───────────────────────┬───────────────────────┘
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  │                                               │
      [ entity1.Priority != entity2.Priority ]       [ entity1.Priority == entity2.Priority ]
                  │                                               │
                  ▼                                               ▼
    ┌───────────────────────────┐                   ┌───────────────────────────┐
    │ return p1 > p2            │                   │ return t1.Before(t2)      │
    │ (Higher priority is Less) │                   │ (Earlier timestamp first) │
    └───────────────────────────┘                   └───────────────────────────┘
```

### 3.1. `Less` (`QueueSortPlugin`)
- **Signature**: `Less(entity1, entity2 fwk.QueuedEntityInfo) bool`
- **Evaluation Logic**:
  ```go
  func (pl *PrioritySort) Less(entity1, entity2 fwk.QueuedEntityInfo) bool {
      p1 := entity1.GetPriority()
      p2 := entity2.GetPriority()
      return (p1 > p2) || (p1 == p2 && entity1.GetTimestamp().Before(entity2.GetTimestamp()))
  }
  ```
- **Heap Invariant**: In Go's container/heap implementation used by `activeQ`, `Less(i, j) == true` indicates that entity `i` should be popped before entity `j`. Therefore, `p1 > p2` yields `true` so that higher priorities sit at the top of the heap.

---

## 4. Key Constants & Status Invariants

| Identifier | Value / Type | Purpose |
| :--- | :--- | :--- |
| `Name` | `names.PrioritySort` (`"PrioritySort"`) | Registered plugin name in scheduler profile. |
| Strict Weak Ordering | `(p1 > p2) || (p1 == p2 && t1 < t2)` | Mathematical ordering ensuring stable heap operation without cycles. |

---

## 5. Architectural Invariants & Edge Cases

1. **Strict Weak Ordering**:
   - Irreflexivity: `Less(a, a)` is always `false`.
   - Asymmetry: If `Less(a, b)` is `true`, `Less(b, a)` is `false`.
   - Transitivity: If `Less(a, b)` and `Less(b, c)` are `true`, then `Less(a, c)` is `true`.
2. **Clock Resolution**: Queue timestamps are recorded via `time.Now()` upon enqueue. When timestamps are identical and priorities are equal, heap placement preserves stable relative ordering.
3. **Single QueueSort Plugin Rule**: The scheduler framework enforces that exactly one `QueueSortPlugin` is enabled across all profiles in a scheduler instance.

---

## 6. Test Fixtures & Unit Testing

`priority_sort_test.go` verifies the comparator logic across all permutations:
- **Priority Differential**: Confirms `highPriority` beats `lowPriority` regardless of arrival time.
- **Equal Priority FIFO**: Confirms earlier `Timestamp` precedes later `Timestamp` when priority is identical.
- **Test Helpers**: Uses `st.MakePod().Priority(...)` and `framework.NewPodInfo` to construct test `QueuedPodInfo` structs with explicit `QueueingParams.Timestamp`.
