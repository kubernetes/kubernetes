# AGENTS.md: Developer & Agent Guide for `pkg/scheduler/framework/plugins/topologyaware`

This guide provides AI agents and human contributors with an architectural overview, lifecycle walkthrough, state management mechanics, performance optimizations, and testing strategies for the `TopologyPlacement` scheduler plugin located at `pkg/scheduler/framework/plugins/topologyaware`.

---

## 1. High-Level Overview & Core Role

The `TopologyPlacement` plugin (`names.TopologyPlacementGenerator`) is a placement generator plugin designed for topology-aware workload and gang scheduling under the Generic Workload scheduling framework (`features.GenericWorkload`, `features.TopologyAwareWorkloadScheduling`, and `features.CompositePodGroup`).

### Key Objectives:
1. **Domain-Constrained Placement Generation**: Generates candidate `fwk.Placement` sets partitioned by topological domain labels (e.g. availability zones, NUMA domains, racks, hostnames) for all member pods of a `PodGroup` or `CompositePodGroup`.
2. **Strict Domain Homogeneity**: Ensures that all pods belonging to a topology-constrained PodGroup land strictly within the same topology domain.
3. **Multi-Pod Group State Tracking**: Consults previously scheduled member pods in the group and forces remaining pods into the same domain already chosen for the group.
4. **Hierarchical Workload Support**: Seamlessly traverses composite pod group trees to resolve topology constraints across composite parent-child relationships.

---

## 2. Implemented Extension Points

`TopologyPlacement` implements the following framework interface:

| Extension Point | Interface | Phase & Execution Mode | Description |
|---|---|---|---|
| **PlacementGenerate** | `fwk.PlacementGeneratePlugin` | PodGroup Scheduling Loop | Generates candidate `Placement` instances by partitioning nodes according to topology constraints. |

---

## 3. Placement Generation Algorithm & Lifecycle

The plugin is executed during the multi-pod / gang scheduling cycle (`schedule_one_podgroup.go`) via `GeneratePlacements`:

```
                             [ Parent Placement ]
                           (Set of Candidate Nodes)
                                      │
                                      ▼
                        Read Topology Constraints
                     from PodGroup / CompositePodGroup
                                      │
               ┌──────────────────────┴──────────────────────┐
               ▼                                             ▼
        No Constraints                             Topology Key Specified
               │                                             │
      Return Parent Placement                                ▼
      (Single Unconstrained Placement)           Query Scheduled Pods in Group
                                                             │
                                              ┌──────────────┴──────────────┐
                                              ▼                             ▼
                                    Pods Already Scheduled         No Pods Scheduled Yet
                                              │                             │
                                  Determine Scheduled Domain         Partition ALL parent
                                  & filter nodes to that            nodes into candidate
                                  single domain                     placements per domain
                                              │                             │
                                              └──────────────┬──────────────┘
                                                             ▼
                                                Emit []fwk.Placement
```

### 3.1. Topology Key Extraction (`getTopologyKey`)
- Inspects `.spec.schedulingConstraints.topology` on `PodGroup` or `CompositePodGroup`.
- Extracts the primary topology key (e.g. `topology.kubernetes.io/zone`, `kubernetes.io/hostname`, `topology.node.kubernetes.io/numa-domain`).
- If no topology constraints are specified, returns the original `parentPlacement` untouched.

### 3.2. Scheduled Domain Resolution (`getScheduledPodsTopologyDomain`)
1. **Retrieve Scheduled Members**: Fetches already scheduled pods in the group via `SnapshotSharedLister().PodGroupStates()` (or iterates over composite sub-groups when `EnableCompositePodGroup` is active).
2. **Domain Consistency Validation**:
   - Queries the node assigned to each scheduled pod from the snapshot.
   - Extracts the node's label matching `topologyKey`.
   - **Invariant**: If member pods are found on nodes in different topology domains (or if a node lacks the topology label), returns `framework.NewStatus(framework.Error, "more than 1 domain found for pod group")`.
   - Sets `requiredDomain` to the identified domain.

### 3.3. Candidate Node Partitioning
- Iterates over all nodes in `parentPlacement.Nodes`.
- Reads `node.Labels[topologyKey]`.
- If `requiredDomain` is set, discards nodes in any other domain.
- Groups surviving nodes into `nodesPerTopologyDomain[domain]`.
- Constructs a discrete `*fwk.Placement` for each valid domain:
  ```go
  placements = append(placements, &fwk.Placement{
      Name:  topologyDomain,
      Nodes: nodes,
  })
  ```

---

## 4. Feature Gates & Workload Hierarchies

The plugin operates across several interconnected Kubernetes feature gates:

| Feature Gate | Default | Role in Plugin |
|---|---|---|
| **`GenericWorkload`** | Alpha/Beta | Enables PodGroup abstractions and multi-pod scheduling loops. |
| **`TopologyAwareWorkloadScheduling`** | Alpha/Beta | Activates `TopologyPlacementGenerator` in the framework registry. |
| **`CompositePodGroup`** | Alpha/Beta | Enables hierarchical pod groups (`CompositePodGroup`), recursive state lookup via `helper.GetPodGroupStates`, and tree-wide topology domain unification. |

---

## 5. Performance Optimizations & Invariants

1. **In-Memory Snapshot Resolution**:
   - Node lookups and PodGroup state queries use `Handle.SnapshotSharedLister()`, executing strictly against in-memory cache snapshots without generating API server queries.
2. **Early Placement Filtering**:
   - By creating discrete placements per topology domain upfront, the scheduling algorithm prunes non-viable nodes in bulk before single-pod filtering and scoring cycles run.
3. **Strict Domain Singletons**:
   - Pod groups cannot span multiple domains when a topology constraint is active. If any member is already placed, `requiredDomain` collapses candidate placements to a single domain immediately.

---

## 6. Testing & Verification Guide

### Unit Tests
Execute unit tests for `topologyaware`:
```bash
cd /home/debian/work
GOTOOLCHAIN=auto go test -v -race ./pkg/scheduler/framework/plugins/topologyaware/...
```

### Key Test Scenarios:
- `TestGeneratePlacements`:
  - PodGroups without topology constraints (returns original placement).
  - PodGroups with topology keys partitioning nodes into distinct domain placements.
  - PodGroups with already-scheduled pods locking subsequent pods into the established domain.
  - Error conditions when already-scheduled pods span conflicting domains or land on unlabeled nodes.
  - Composite pod group hierarchies across both `CompositePodGroup` enabled and disabled modes.
