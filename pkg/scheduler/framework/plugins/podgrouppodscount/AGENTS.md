# PodGroupPodsCount Plugin

This guide provides an architectural overview, interface implementations, placement scoring formulas, score normalization algorithms, composite hierarchy aggregation, and testing strategies for the `PodGroupPodsCount` plugin in `pkg/scheduler/framework/plugins/podgrouppodscount`.

---

## 1. High-Level Purpose & Scope

The `PodGroupPodsCount` plugin is a **`PlacementScorePlugin`** designed for multi-pod and gang scheduling workflows. It prioritizes multi-node placements that maximize the total number of pods accommodated from a considered `PodGroup` (or hierarchical `CompositePodGroup`).

### Core Responsibilities:
1. **Placement Prioritization**: Evaluates candidate placements during PodGroup placement cycles, scoring each candidate according to how many group pods are accommodated (both previously assigned/assumed pods and newly proposed assignments).
2. **Smooth Scoring Curve**: Integrates existing scheduled pods with proposed placements into the raw score, reducing relative score volatility between candidate options.
3. **Hierarchical Aggregation**: When `EnableCompositePodGroup` is enabled, aggregates scheduled pod counts across all child pod groups in the composite hierarchy tree.
4. **Normalized Score Distribution**: Normalizes raw placement scores to the standard framework range `[fwk.MinScore, fwk.MaxScore]` (`[0, 100]`), intentionally omitting `MinCount` to avoid artificial score cliffs.

---

## 2. Package Architecture & File Map

```
pkg/scheduler/framework/plugins/podgrouppodscount/
├── podgroup_pods_count.go       # Plugin definition, ScorePlacement, NormalizePlacementScore, and count aggregation
├── podgroup_pods_count_test.go  # Unit tests covering ScorePlacement and NormalizePlacementScore
└── AGENTS.md                    # This agent documentation
```

---

## 3. Data Structures & Plugin Configuration

### 3.1. `PodGroupPodsCount` Struct

```go
type PodGroupPodsCount struct {
    handle fwk.Handle
    fts    feature.Features
}
```

- **`handle`**: Framework handle providing access to shared lister snapshots (`SnapshotSharedLister()`) to retrieve `PodGroupState` metrics.
- **`fts`**: Feature gates structure checking `fts.EnableCompositePodGroup`.

### 3.2. Constants

| Constant | Value | Purpose |
| :--- | :--- | :--- |
| `Name` | `names.PodGroupPodsCount` (`"PodGroupPodsCount"`) | Registered plugin name in scheduler configuration. |

---

## 4. Extension Point Implementations

`PodGroupPodsCount` implements `fwk.PlacementScorePlugin` and `fwk.PlacementScoreExtensions`.

```
              ┌─────────────────────────────────────────────────────────┐
              │                   ScorePlacement Cycle                  │
              │         (Evaluate Candidate Node Placement)             │
              └────────────────────────────┬────────────────────────────┘
                                           │
                                           ▼
              ┌─────────────────────────────────────────────────────────┐
              │ Count Already Scheduled Pods (Assumed + Assigned):      │
              │ - Flat PodGroup: pgState.ScheduledPodsCount()           │
              │ - Composite: Sum(child.ScheduledPodsCount())            │
              └────────────────────────────┬────────────────────────────┘
                                           │
                                           ▼
              ┌─────────────────────────────────────────────────────────┐
              │ Compute Raw Placement Score:                            │
              │ RawScore = ScheduledCount + len(ProposedAssignments)   │
              └────────────────────────────┬────────────────────────────┘
                                           │
                                           ▼
              ┌─────────────────────────────────────────────────────────┐
              │ NormalizePlacementScore:                                │
              │ Score = MinScore + (RawScore * (MaxScore - MinScore))   │
              │                  / maxCount                             │
              └─────────────────────────────────────────────────────────┘
```

### 4.1. `ScorePlacement` (`PlacementScorePlugin`)

- **Signature**: `ScorePlacement(ctx context.Context, state fwk.PlacementCycleState, podGroup fwk.PodGroupInfo, placement *fwk.PodGroupAssignments) (int64, *fwk.Status)`
- **Calculation Formula**:
  $$\text{RawScore} = \text{ScheduledPodsCount} + |\text{ProposedAssignments}|$$
- **Rationale**:
  Including already-scheduled pods ensures that small incremental differences in placement proposals produce proportionally gradual variations in placement scores, rather than drastic step changes.
- **Hierarchy Aggregation (`getScheduledPodsCount`)**:
  - **Flat PodGroup** (`!fts.EnableCompositePodGroup`): Retrieves `PodGroupState` for `(namespace, name)` from the snapshot lister and returns `pgState.ScheduledPodsCount()`.
  - **Composite PodGroup** (`fts.EnableCompositePodGroup`): Iterates over all child pod group states via `helper.GetPodGroupStates(snapshotLister, podGroup.GetKey())` and sums their `ScheduledPodsCount()`.

### 4.2. `NormalizePlacementScore` (`PlacementScoreExtensions`)

- **Signature**: `NormalizePlacementScore(ctx context.Context, state fwk.PodGroupCycleState, podGroup fwk.PodGroupInfo, scores []fwk.PlacementScore) *fwk.Status`
- **Normalization Formula**:
  $$\text{NormalizedScore}_i = \text{MinScore} + \frac{\text{Score}_i \times (\text{MaxScore} - \text{MinScore})}{\text{maxCount}}$$
  where:
  $$\text{maxCount} = \max_{j}(\text{Score}_j)$$
- **Zero-Max Guard**:
  If $\text{maxCount} == 0$, returns an error status (`"no pods from pod group are assigned to any of the candidate placements"`). In standard operation, upstream gang filtering (`GangScheduling`) prevents placements with zero schedulable pods from reaching the scoring stage.
- **Design Invariant (Omission of `MinCount`)**:
  Normalization strictly divides by $\text{maxCount}$ across candidate placements without shifting by `MinCount`. Shifting or rescaling relative to `MinCount` would introduce artificial score cliffs between candidate placements that differ by only a single pod assignment.

---

## 5. Integration with the PodGroup Placement Cycle

```
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ 1. Filter Phase: Identify viable node subsets across the cluster         │
  └────────────────────────────────────┬─────────────────────────────────────┘
                                       │
                                       ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ 2. Placement Generation: Form candidate multi-node placement topologies  │
  └────────────────────────────────────┬─────────────────────────────────────┘
                                       │
                                       ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ 3. ScorePlacement: Score each candidate placement                        │
  │    - PodGroupPodsCount scores higher for placements fitting more pods    │
  │    - Other placement scorers (e.g. topology spread) add dimensions       │
  └────────────────────────────────────┬─────────────────────────────────────┘
                                       │
                                       ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ 4. NormalizePlacementScore: Scale raw counts to [MinScore, MaxScore]     │
  └────────────────────────────────────┬─────────────────────────────────────┘
                                       │
                                       ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ 5. Best Placement Selection: Highest aggregate score candidate selected  │
  └──────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Testing Strategy & Test Coverage

The unit test suite in `podgroup_pods_count_test.go` verifies:

1. **`TestScorePlacement`**:
   - Validates correct addition of scheduled pod count + proposed assignment count.
   - Tests flat `PodGroup` with 0, 1, and N scheduled pods.
   - Tests hierarchical `CompositePodGroup` with multiple sub-groups and nested assigned pods.
   - Tests error handling when snapshot state lookup fails.
2. **`TestNormalizePlacementScore`**:
   - Validates scaling from raw scores to `[0, 100]`.
   - Tests single placement candidate normalization (maxCount scales to 100).
   - Tests multi-candidate score distribution (e.g. counts `[2, 4]` scale to `[50, 100]`).
   - Asserts error status when all candidates have `Score == 0`.
