# Domain 5: Advanced Resource, Topology & Batched Preemption

**Governing KEPs:**
- **KEP-2837**: *Pod-Level Resource Specification*
- **KEP-5598**: *Opportunistic Batching & Pod Signatures*
- **KEP-4818**: *Node Declared Features*
- **KEP-6072**: *Structured Dynamic Resource Allocation (DRA) & NUMA Topology Alignment*

**Primary Packages:**
- `pkg/scheduler/framework/plugins/noderesources/`
- `pkg/scheduler/framework/plugins/nodedeclaredfeatures/`
- `pkg/scheduler/framework/plugins/dynamicresources/`
- `pkg/scheduler/backend/queue/`
- `staging/src/k8s.io/kube-scheduler/framework/`

---

## 1. Executive Summary & Problem Statement

As modern workloads diversify into specialized AI accelerators (GPUs, TPUs, NPUs), high-speed interconnects (RDMA, InfiniBand), NUMA-aligned multi-socket systems, and massive replica sets (10,000+ pods), preemption logic can no longer rely purely on simple scalar CPU/memory sums.

This domain covers four critical enhancement proposals extending preemption to complex topologies, hardware architectures, and extreme scale:
1. **Pod-Level Resources (KEP-2837)**: Reconciling container-level resource sums vs pod-level explicit requests and overheads during preemption calculations.
2. **Opportunistic Batching & Pod Signatures (KEP-5598)**: Reusing preemption candidate evaluations and node simulations across identical pods using cryptographic pod signatures to avoid redundant PostFilter cycles.
3. **Node Declared Features (KEP-4818)**: Matching inferred CPU microarchitectures, instruction sets (AVX-512, AMX), and hardware capabilities before considering candidate preemption nodes.
4. **Structured DRA & NUMA Alignment (KEP-6072)**: Preemption and de-allocation of hardware device claims across NUMA domains without corrupting device driver states.

---

## 2. Technical Details by Enhancement

### 2.1 Pod-Level Resources (KEP-2837)
When `spec.resources` is defined at the Pod level (or when `spec.overhead` via `RuntimeClass` is present), the effective resource request $R_{\text{effective}}$ for preemption calculation is computed as:

$$R_{\text{effective}} = \max\left( \text{Pod.Spec.Resources.Requests}, \sum_{c \in \text{Containers}} c.\text{Resources.Requests} \right) + \text{Pod.Spec.Overhead}$$

- **Preemption Accounting**: When evaluating whether a preemptor fits or how much capacity a victim releases, the preemption engine uses $R_{\text{effective}}$ instead of raw container sums.
- **Overhead Accounting**: Guarantees that sandboxed container runtimes (e.g., Kata Containers, gVisor) reserve memory for VM management, preventing node OOM panics post-preemption.

### 2.2 Opportunistic Batching & Pod Signatures (KEP-5598)
In large deployments, hundreds of identical pods arrive simultaneously in `activeQ`.
- **Pod Signature Calculation**: A cryptographic SHA-256 hash is computed over all scheduling-relevant fields (`Spec.Affinity`, `Spec.Tolerations`, `Spec.NodeSelector`, `Spec.Resources`, `Priority`, `TopologySpreadConstraints`).
- **Signature Reuse**: If Pod 1 of a signature $S$ executes `PostFilter` and determines that Node $N$ requires evicting victims $\{V_1, V_2\}$, subsequent pods with signature $S$ in the same batch reuse the candidate evaluation without repeating full cluster-wide filter simulations.

### 2.3 Node Declared Features (KEP-4818)
- Nodes declare specialized hardware capabilities in `node.Status.DeclaredFeatures` (e.g., `intel.com/avx512=true`, `arm.com/sve2=true`).
- **Preemption Filtering**: During preemption candidate discovery, candidate nodes must satisfy declared feature predicates. The scheduler will not evict lower-priority pods on a node that lacks the required hardware feature, avoiding futile evictions.

### 2.4 Structured DRA & NUMA Alignment (KEP-6072)
Dynamic Resource Allocation (DRA) manages fine-grained hardware devices:
- **ResourceClaim Preemption**: When a high-priority pod requires a device allocated to a lower-priority pod's `ResourceClaim`, the scheduler evaluates device claim revocations.
- **NUMA Domain Co-Locality**: Ensures that preempting a victim frees device claims and memory within the exact NUMA domain required by the preemptor.

---

## 3. Subsystem Integration & Invariants

```
+-----------------------------------------------------------------------------+
|                          Advanced Preemption Flow                           |
+-----------------------------------------------------------------------------+
                                       |
                   [Compute Pod Signature (KEP-5598)]
                                       |
                   [Calculate Effective Resources (KEP-2837)]
                   (Pod Requests + Overhead)
                                       |
                   [Filter Candidate Nodes with Declared Features (KEP-4818)]
                                       |
                   [Evaluate DRA Device Claims & NUMA Domains (KEP-6072)]
                                       |
                   [Execute Actuation & Update Cache]
```

---

## 4. Verification & Test Matrix

- **Unit Tests**:
  - `pkg/scheduler/framework/plugins/noderesources/resource_allocation_test.go`
  - `pkg/scheduler/framework/plugins/nodedeclaredfeatures/node_declared_features_test.go`
  - `staging/src/k8s.io/dynamic-resource-allocation/structured/internal/allocatortesting/`
- **Integration Tests**: `test/integration/scheduler/preemption/`
  - `TestPodOverheadPreemptionFit`: Validates runtime overhead integration.
  - `TestDRAClaimPreemption`: Validates structured device allocation preemption.
