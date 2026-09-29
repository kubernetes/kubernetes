# Technical Report: Kubernetes Preemption Test Infrastructure, Flake Remediation, and Performance Benchmarks

## Executive Summary

The Kubernetes preemption subsystem has seen rapid architectural evolution, including the delivery of **Workload-Aware Preemption (WAP) / Gang Scheduling (KEP-5710)**, **CompositePodGroups (KEP-6012)**, **Asynchronous Preemption**, **Scheduler Async API Calls**, and **In-Place Pod Vertical Scaling Preemption (KEP-1287)**. 

As these capabilities expanded the state space of `kube-scheduler`, the integration and end-to-end (e2e) test harnesses faced severe scalability and reliability challenges:
1. **Package Timeout Exhaustion**: Cartesian feature gate evaluation combined with monolithic test packages repeatedly exceeded the 600-second integration test deadline (`KUBE_TIMEOUT`), triggering context cancellations that tore down shared API servers and caused cascading "connection refused" failures.
2. **Setup Overhead & Redundant Initialization**: Instantiating isolated `kube-apiserver` and `etcd` instances for every subtest within cartesian test matrices caused excessive runtime overhead (~360s for a single test suite).
3. **Concurrency and Race Hazards in Test Hooks**: Custom framework plugins (e.g., `Permit` and `PreBind` interception plugins) suffered from unsynchronized memory access and premature assertions prior to pod binding completion.
4. **Inter-Test Resource Contention**: In parallel e2e suites, cluster-wide extended resources and node volume attachment slots were subject to cross-test pollution and race conditions, causing spurious evictions of unrelated test workloads.
5. **Benchmarking Complexity**: Schedulers evaluating gang preemption across multi-zone topology spread constraints and multi-victim disruption modes required standardized benchmarking scenarios, realistic resource ratios, and normalized priority hierarchies.

This report provides a comprehensive technical investigation of the root causes, architectural fixes, performance optimizations, and benchmark methodologies implemented across recent Kubernetes preemption pull requests.

---

## 1. Summary of Investigated Pull Requests & Remediations

| Pull Request | Branch / Author | Component & File(s) | Primary Root Cause / Issue | Technical Resolution & Impact |
| :--- | :--- | :--- | :--- | :--- |
| **PR #141048** | `BenTheElder/preemption-split-podgroup` | `test/integration/scheduler/preemption/` | Monolithic preemption package exceeded 600s `KUBE_TIMEOUT`, causing cascading context cancellation. | Split PodGroup preemption tests into standalone package `test/integration/scheduler/preemption/podgroup`. |
| **PR #140737** | `shwetha-s-poojary/fix_flake_TestPreemption` | `test/integration/scheduler/preemption/preemption_test.go`, `test/integration/util/util.go` | Instantiating 72 separate API servers across feature gate permutations took ~360s, exhausting CI timeouts on slower architectures (e.g., ppc64le). | Implemented `WithNewNamespace` helper and grouped subtests by `GenericWorkload`; reduced API server startups from 72 to 8, cutting execution to ~70s (~80% reduction). |
| **PR #140408** | `GFilipek/preemptionperf` | `test/integration/scheduler_perf/workload_preemption/` | Lack of gang preemption benchmarks under topology spread constraints; inconsistent priority values across templates. | Added `GangPreemptionTopologySpreading` benchmark scenario; normalized priority values to uniform standard (`10000`). |
| **PR #140651** | `vshkrabkov/def-preemption-perf-test-extend` | `test/integration/scheduler_perf/default_preemption/` | Missing comparative performance benchmarks evaluating single-pod preemption against PodGroup victims with varying disruption modes. | Introduced `PreemptionPodGroupDisruptionMode` benchmark comparing `NoPodGroup`, `DisruptionSingle`, and `DisruptionAll` using a calibrated 10:1 victim-to-preemptor ratio. |
| **PR #140872** | `pacoxu/fix-wap-flake` | `test/e2e/scheduling/workload_aware_preemption.go` | Hardcoded cluster-wide extended resource name (`example.com/combined-resource`) caused race conditions and resource contention across parallel e2e tests. | Dynamically scoped extended resource names to test namespaces (`example.com/<namespace>`), ensuring complete isolation. |
| **PR #138017** | `srivastav-abhishek/preemption-test-fix` | `test/integration/scheduler/preemption/preemption_test.go` | Race conditions in plugin state initialization and missing synchronization on preemptor scheduling before asserting node placement. | Initialized plugin dispatch maps prior to scheduler startup; added explicit polling for `PodScheduled=True` condition. |
| **PR #135623** | `jsafrane/fix-preemption-e2e` | `test/e2e/storage/testsuites/readwriteoncepod.go` | Parallel e2e tests hijacked node volume attachment slots freed during RWOP preemption, causing secondary eviction of unrelated test pods. | Annotated ReadWriteOncePod (RWOP) preemption e2e tests with `f.WithSerial()` / `ginkgo.Serial` execution guards. |
| **PR #135372** | `ingvagabund/e2e-scheduler-preemption-async-fix` | `test/e2e/scheduling/preemption.go` | Concurrent submission of high- and medium-priority pods allowed high-priority pods to steal spots, preventing medium-priority pods from triggering preemption. | Staged pod creation sequentially: created all medium-priority pods and verified low-priority pod eviction before creating high-priority pods. |

---

## 2. Suite Splitting & Timeout Exhaustion Mitigation

### 2.1 Context and Root Cause (PR #141048)

In the Kubernetes integration test framework, tests are executed per Go package via `hack/make-rules/test-integration.sh`. The integration harness enforces a strict flat package timeout defined by `KUBE_TIMEOUT` (default: 600 seconds / 10 minutes).

With the introduction of Workload-Aware Preemption, Gang Scheduling, and Composite PodGroups, the `test/integration/scheduler/preemption` package expanded rapidly. In addition to testing standard pod preemption, the suite incorporated comprehensive test matrices for PodGroup workflows.

Prior to PR #141048, the package runtime was dominated by PodGroup integration tests:

```text
TestPodGroupPreemption                           166.5s
TestCompositePodGroupPreemption                  100.6s
TestPodGroupPreemption_NominatedNodeNameRespected  7.0s
TestPodGroupCycleStatePreserved                    4.1s
TestPodGroupPreemptionStatus                       3.7s
Total PodGroup Subtest Execution Time:          ~281.9s
```

Combined with `TestPreemption` (evaluating permutations of `SchedulerAsyncPreemption`, `SchedulerAsyncAPICalls`, `ClearingNominatedNodeNameAfterBinding`, and `GenericWorkload`), `TestDeferredResizePreemption`, and other subtests, the cumulative package execution time routinely approached or exceeded 600 seconds on standard CI runners.

When the 600-second deadline expired:
1. `ktesting` received a timeout signal and canceled the package context (`ctx.Done()`).
2. The package-level `TestMain` or shared test harness began tearing down etcd and API server instances.
3. In-flight subtests executing subsequent cases suffered immediate socket teardowns, failing with `connection refused` or `context canceled`.
4. Whichever subtest happened to execute last in Go's pseudo-random subtest ordering was marked as failed, masking the real culprit (package-level timeout exhaustion) and making CI failure triage difficult.

### 2.2 Architectural Resolution

PR #141048 extracted all PodGroup-specific preemption tests from `test/integration/scheduler/preemption` into an independent subpackage:

```text
test/integration/scheduler/preemption/
├── asyncframework/
├── deferred_resize_preemption_test.go
├── main_test.go
├── misc/
├── nominatednodename/
├── podgroup/                         <-- Standalone Package (PR #141048)
│   ├── main_test.go
│   └── podgrouppreemption_test.go
└── preemption_test.go
```

Key aspects of this separation:
* **Dedicated Timeout Budget**: By residing in `test/integration/scheduler/preemption/podgroup`, the PodGroup tests receive their own dedicated 600-second `KUBE_TIMEOUT` allocation.
* **Isolated `TestMain` & Etcd Lifecycle**: `podgroup/main_test.go` initializes its own etcd instance via `framework.EtcdMain(m.Run)`, completely decoupling its storage and server lifecycle from parent package executions.
* **Symbol Encapsulation**: PodGroup preemption tests were fully self-contained and shared no unexported symbols with `preemption_test.go`, allowing a clean package declaration (`package podgrouppreemption`) without duplicated test utilities.
* **Architectural Consistency**: This structure mirrors previously partitioned scheduler subpackages, including `preemption/misc` and `preemption/nominatednodename`.

---

## 3. Test Performance Optimization & Shared API Server Infrastructure

### 3.1 Overhead in Cartesian Feature Gate Testing (PR #140737)

`TestPreemption` in `test/integration/scheduler/preemption/preemption_test.go` evaluates whether scheduler preemption functions correctly across combinations of underlying flags and feature gates. The test matrix evaluated:
* `features.SchedulerAsyncPreemption`: `[true, false]` (2 states)
* `features.SchedulerAsyncAPICalls`: `[true, false]` (2 states)
* `features.ClearingNominatedNodeNameAfterBinding`: `[true, false]` (2 states)
* `features.GenericWorkload`: `[true, false]` (2 states across 9 base test scenarios)

Originally, `TestPreemption` instantiated a new API server and etcd instance for every individual subtest case. This resulted in:
$$\text{Total API Server Startups} = 2 \times 2 \times 2 \times 9 = 72 \text{ startups}$$

Each API server initialization entails generating certificates, binding loopback TCP ports, establishing OpenAPI schemas, starting CRD controllers, and establishing etcd connections, requiring 3 to 6 seconds per instance. On resource-constrained architectures (such as `ppc64le`, `s390x`, or overloaded cloud VMs), total test runtime reached ~360 seconds.

### 3.2 Shared API Server and `WithNewNamespace` Mechanism

PR #140737 introduced a lifecycle pattern where a single API server is shared across multiple subtests, while maintaining strict isolation at the namespace layer.

#### Helper Implementation (`test/integration/util/util.go`)

```go
// WithNewNamespace creates a child TestContext that shares the API server from
// parent but gets a fresh namespace. Only the namespace is deleted on t.Cleanup;
// the API server lifecycle is managed by the caller. This is useful when
// multiple subtests share one API server to avoid the per-subtest startup cost.
func WithNewNamespace(t *testing.T, parent *TestContext, nsPrefix string) *TestContext {
	t.Helper()
	child := &TestContext{
		ClientSet:  parent.ClientSet,
		KubeConfig: parent.KubeConfig,
		Ctx:        parent.Ctx,
		CloseFn:    func() {}, // API server is owned by parent; do not tear it down here.
	}
	child.NS = framework.CreateNamespaceOrDie(child.ClientSet, nsPrefix+string(uuid.NewUUID()), t)
	t.Cleanup(func() {
		framework.DeleteNamespaceOrDie(child.ClientSet, child.NS, t)
	})
	return child
}
```

#### Feature Gate Grouping & Subtest Matrix

To ensure that the API server's static feature gate configuration strictly matches the scheduler's feature gates, tests are partitioned upfront by `genericWorkloadEnabled`:

```go
// Group test indexes by genericWorkloadEnabled so each group can share an
// API server started with the correct feature gate value.
testsByGW := make(map[bool][]int)
for i, test := range tests {
    testsByGW[test.genericWorkloadEnabled] = append(testsByGW[test.genericWorkloadEnabled], i)
}

for _, asyncPreemptionEnabled := range []bool{true, false} {
    for _, asyncAPICallsEnabled := range []bool{true, false} {
        for _, clearingNominatedNodeNameAfterBinding := range []bool{true, false} {
            for _, genericWorkloadEnabled := range []bool{true, false} {
                gwIndexes := testsByGW[genericWorkloadEnabled]
                if len(gwIndexes) == 0 {
                    continue
                }
                // One API server per full flag combination.
                featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
                    features.SchedulerAsyncPreemption:              asyncPreemptionEnabled,
                    features.SchedulerAsyncAPICalls:                asyncAPICallsEnabled,
                    features.ClearingNominatedNodeNameAfterBinding: clearingNominatedNodeNameAfterBinding,
                    features.GenericWorkload:                       genericWorkloadEnabled,
                })
                sharedAPICtx := testutils.InitTestAPIServer(t, "preemption", nil)

                for _, i := range gwIndexes {
                    test := tests[i]
                    t.Run(fmt.Sprintf("%s (...)", test.name), func(t *testing.T) {
                        testCtx := testutils.InitTestSchedulerWithOptions(t,
                            testutils.WithNewNamespace(t, sharedAPICtx, "preemption"),
                            0,
                            scheduler.WithProfiles(cfg.Profiles...),
                            scheduler.WithFrameworkOutOfTreeRegistry(registry))
                        testutils.SyncSchedulerInformerFactory(testCtx)
                        go testCtx.Scheduler.Run(testCtx.SchedulerCtx)
                        defer testCtx.SchedulerCloseFn()

                        // Subtest logic, object creation, assertions, and per-test pod/node cleanup
                        ...
                    })
                }
            }
        }
    }
}
```

### 3.3 Performance Impact

* **Startup Count Reduction**: Decreased from **72** down to **8** (a $9 \times$ reduction in API server lifecycles).
* **Execution Duration**: Total runtime dropped from **~360 seconds to ~70 seconds** (an ~80% performance improvement).
* **Flake Elimination**: Completely eliminated `TestAsyncPreemption` flakes caused by context deadline timeouts on slower CI architectures.

---

## 4. Performance Benchmarking Scenarios & Methodologies

The Kubernetes scheduler performance benchmark suite (`test/integration/scheduler_perf`) provides declarative benchmarking configs evaluated with `scheduler_perf` runner. Recent PRs introduced realistic multi-dimensional gang preemption scenarios and structured benchmark suites.

### 4.1 Topology Spreading Gang Preemption (PR #140408)

Gang scheduling (scheduling an entire `PodGroup` atomically) combined with multi-zone `topologySpreadConstraints` presents a severe computational challenge for the scheduler's preemption plugin:
1. The scheduler must identify candidate nodes capable of accommodating member pods.
2. The candidate nodes must satisfy zone-level skew limits (`maxSkew: 1`).
3. If preemption is required, victim selection must not violate topology spreading for existing workloads or the incoming gang.

PR #140408 introduced the `GangPreemptionTopologySpreading` benchmark under `test/integration/scheduler_perf/workload_preemption/performance-config.yaml`:

```yaml
- name: GangPreemptionTopologySpreading
  featureGates:
    GenericWorkload: true
  workloadTemplate:
  - opcode: createNodes
    countParam: $initNodes
    nodeTemplatePath: ../templates/node-default.yaml
    labelNodePrepareStrategy:
      labelKey: "topology.kubernetes.io/zone"
      labelValues: ["zone-1", "zone-2", "zone-3"]
  - opcode: createNamespaces
    prefix: gang-preempt
    count: 2
  - opcode: createPods
    countParam: $initPods
    podTemplatePath: templates/pod-low-priority.yaml
    namespace: gang-preempt-1
    templateParams:
      nodesCount: $initNodes
      podsCount: $initPods
  - opcode: createPodGroups
    countParam: $preemptorPodGroups
    namespace: gang-preempt-0
    templatePath: templates/podgroup.yaml
    templateParams:
      podsPerGroup: $podsPerGroup
      priority: 10000
  - opcode: createPods
    collectMetrics: true
    countParam: $preemptorPodGroups
    countMultiplierParam: $podsPerGroup
    namespace: gang-preempt-0
    podTemplatePath: templates/gang-pod-with-topology-spreading.yaml
    templateParams:
      podsPerGroup: $podsPerGroup
      cpuRequest: $gangCpuRequest
      priority: 10000
```

#### Benchmark Workload Matrix
* **Short Integration Workload (`3Nodes_1PreemptorGang_3Pods_Evicting_12InitPods`)**:
  * 3 nodes distributed across `zone-1`, `zone-2`, `zone-3`.
  * 12 initial low-priority pods saturating CPU capacity.
  * 1 preemptor PodGroup with 3 pods requesting 3900m CPU each with `topologySpreadConstraints` (`maxSkew: 1`, `topologyKey: topology.kubernetes.io/zone`, `whenUnsatisfiable: DoNotSchedule`).
* **Scale Performance Workload (`100Nodes_6PreemptorGangs_16Pods_Evicting_1000InitPods`)**:
  * 100 nodes distributed across 3 zones.
  * 1000 initial low-priority saturation pods.
  * 6 preemptor PodGroups of 16 pods each (total 96 gang pods requesting 3970m CPU each).

#### Priority Normalization Fix
Commit `9d21b8ec688` resolved an issue where disparate templates used conflicting hardcoded priority integers (such as `9999999` and `8888888`), causing unintended preemption between test components. Priority values across templates (`gang-pod.yaml`, `gang-pod-with-topology-spreading.yaml`, `pod-anchor-high-priority.yaml`, `podgroup.yaml`, `pod-high-priority.yaml`) were normalized to a standard `10000`.

### 4.2 Pod Group as Victim & Disruption Modes (PR #140651)

PR #140651 restructured preemption benchmarks by creating a dedicated `default_preemption` suite and introducing `PreemptionPodGroupDisruptionMode`.

This benchmark systematically compares preemption latency and throughput across three distinct victim disruption configurations:
1. **`NoPodGroup`**: Low-priority victim pods are standalone pods (no `PodGroup` association).
2. **`PodGroup_DisruptionSingle` (`disruptionMode: "single"`)**: Low-priority victim pods belong to PodGroups configured with `single` disruption mode. The scheduler evicts only the minimal number of victim pods necessary to accommodate the preemptor, leaving the remainder of the victim gang running.
3. **`PodGroup_DisruptionAll` (`disruptionMode: "all"`)**: Low-priority victim pods belong to PodGroups configured with `all` disruption mode. Preempting any member pod forces the immediate eviction of all member pods across the entire cluster.

#### Mathematical Ratio Design (10:1 Victim-to-Preemptor Ratio)

The benchmark is calibrated with the following parameters:
* **Node Capacity**: Standard nodes with 4000m allocatable CPU.
* **Initial State**: 1000 nodes populated with 10,000 low-priority victim pods (10 pods per node). In PodGroup modes, these correspond to 1000 PodGroups (10 pods/group).
* **Victim Sizing**: Each low-priority pod requests 300m CPU (total node allocation: $10 \times 300\text{m} = 3000\text{m}$, leaving 1000m free headroom).
* **Preemptor Request**: High-priority preemptor pods request 3970m CPU each.
* **Eviction Arithmetic**:
  $$\text{Required CPU} = 3970\text{m}$$
  $$\text{Free Headroom} = 4000\text{m} - 3000\text{m} = 1000\text{m}$$
  $$\text{Deficit to Reclaim} = 3970\text{m} - 1000\text{m} = 2970\text{m}$$
  $$\text{Pods Evicted} = \left\lceil \frac{2970\text{m}}{300\text{m}} \right\rceil = 10 \text{ victim pods}$$

Every preemptor pod requires the eviction of exactly 10 low-priority pods on that node, providing a deterministic $10:1$ eviction ratio to benchmark preemption overhead.

```
+-----------------------------------------------------------------------------------+
| Node Capacity: 4000m CPU                                                          |
+-------------------------------------------------------------+---------------------+
| 10 Low-Priority Victim Pods (10 x 300m = 3000m)             | Free Space (1000m)  |
+-------------------------------------------------------------+---------------------+
                                 |
                                 v  Incoming Preemptor Pod Requests 3970m CPU
+-----------------------------------------------------------------------------------+
| 1 High-Priority Preemptor Pod (3970m CPU)                           | Free (30m)  |
+---------------------------------------------------------------------+-------------+
| Result: All 10 Low-Priority Victim Pods evicted on the node (10:1 victim ratio)    |
+-----------------------------------------------------------------------------------+
```

---

## 5. Flake Fixes, Race Conditions, and Concurrency Guards

### 5.1 Per-Namespace Extended Resource Isolation (PR #140872)

#### Root Cause
In `test/e2e/scheduling/workload_aware_preemption.go`, Workload-Aware Preemption e2e tests tested custom resource preemption by registering an extended resource on cluster nodes using a hardcoded global name:
```go
const extendedResourceName = "example.com/combined-resource"
```

When Kubernetes e2e tests execute in parallel mode (`ginkgo -p`), multiple test processes running concurrently in separate namespaces registered and deleted `example.com/combined-resource` on the same shared cluster nodes. One test suite would mutate node capacity or consume extended resource units while another test was expecting exclusive allocation, causing non-deterministic scheduling failures and flake reports.

#### Resolution
PR #140872 dynamically scoped the extended resource name using the unique generated test namespace:
```go
const extendedResourceDomain = "example.com/"
...
extendedResourceName := v1.ResourceName(extendedResourceDomain + ns)
```
This guarantees that each parallel test suite operates on a unique extended resource key (e.g., `example.com/e2e-wap-4921`), completely isolating node resource modifications.

### 5.2 Mutex Protection and Synchronization in Integration Tests (PR #138017)

#### Root Cause
In `test/integration/scheduler/preemption/preemption_test.go`:
1. `TestPreemptionRespectsWaitingPod` and `TestPreemptionRespectsBindingPod` utilized mock plugins (`blockingPermitPlugin` and `blockingPreBindPlugin`) to halt victim pods during specific scheduling framework extension points.
2. In the initial test setup, the control maps (`podsToBlock` and `podToChannels`) were mutated directly on the plugin struct *after* the scheduler background goroutine was already running (`go testCtx.Scheduler.Run(...)`), causing concurrent read/write data races.
3. In `TestPreemptionRespectsBindingPod`, the test checked final pod node placements (`v.Spec.NodeName` and `p.Spec.NodeName`) immediately after unblocking the victim, without awaiting the preemptor pod's binding phase. Because pod binding is asynchronous in `kube-scheduler`, the assertions executed before the preemptor was bound, intermittently reading empty node names.

#### Resolution
1. **Pre-instantiation Dispatch Maps**: Control maps are now instantiated upfront and passed directly into plugin factory closures before registering with the out-of-tree registry:
   ```go
   victimBlockingPlugin := &perPodBlockingPlugin{
       shouldBlock: true,
       blocked:     make(chan struct{}),
       released:    make(chan struct{}),
   }
   podToChannels := map[string]*perPodBlockingPlugin{
       victim.Name: victimBlockingPlugin,
       preemptor.Name: {
           shouldBlock: false,
           blocked:     make(chan struct{}),
           released:    make(chan struct{}),
       },
   }
   registry.Register(blockingPreBindPluginName, func(ctx context.Context, obj runtime.Object, fh fwk.Handle) (fwk.Plugin, error) {
       return newBlockingPreBindPlugin(ctx, obj, fh, podToChannels)
   })
   ```
2. **Explicit Polling for Preemptor Scheduling**: Added polling on the preemptor pod condition `PodScheduled == True` before verifying node placement invariants:
   ```go
   err = wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 10*time.Second, false, func(ctx context.Context) (bool, error) {
       p, err := cs.CoreV1().Pods(testCtx.NS.Name).Get(ctx, preemptor.Name, metav1.GetOptions{})
       if err != nil {
           return false, err
       }
       _, cond := podutil.GetPodCondition(&p.Status, v1.PodScheduled)
       return cond != nil && cond.Status == v1.ConditionTrue, nil
   })
   ```

### 5.3 Storage Preemption Isolation & Serial Execution (PR #135623)

#### Root Cause
In `test/e2e/storage/testsuites/readwriteoncepod.go`, the test `"should preempt lower priority pods using ReadWriteOncePod volumes"` validated that a higher-priority pod can preempt a lower-priority pod holding a `ReadWriteOncePod` (RWOP) PersistentVolumeClaim.

When running in standard parallel e2e suites:
1. Low-priority pod $P_1$ holds the RWOP volume on Node $N$.
2. High-priority pod $P_2$ requests the RWOP volume; scheduler evicts $P_1$.
3. Evicting $P_1$ frees the volume lock and simultaneously frees a volume attachment slot on Node $N$.
4. An unrelated concurrent test pod $P_{\text{unrelated}}$ in another namespace is scheduled and consumes the newly freed volume attachment slot on Node $N$.
5. When $P_2$ attempts to bind to Node $N$, the node attachment limit is exceeded. The scheduler initiates a secondary round of preemption, unexpectedly evicting $P_{\text{unrelated}}$ or failing $P_2$.

#### Resolution
Annotated the RWOP preemption test with `f.WithSerial()` (`ginkgo.Serial`), ensuring that volume preemption tests run in isolation without parallel volume attachment competition.

### 5.4 Priority Inversion & Staging Order in Async Preemption E2E (PR #135372)

#### Root Cause
In `test/e2e/scheduling/preemption.go`, the `SchedulerPreemption` async preemption test scenario created a three-tier priority hierarchy: low-priority pods ($P_{\text{low}}$), medium-priority pods ($P_{\text{med}}$), and high-priority pods ($P_{\text{high}}$).

The test intended to verify that when $P_{\text{med}}$ pods arrive, they trigger preemption of all $P_{\text{low}}$ pods, and subsequent $P_{\text{high}}$ pods preempt or prioritize over $P_{\text{med}}$ pods.

However, the original test logic created $P_{\text{med}}$ and $P_{\text{high}}$ pods concurrently in the same loop:
* If $P_{\text{high}}$ pods were processed by the scheduler before all $P_{\text{med}}$ pods were evaluated, $P_{\text{high}}$ pods directly claimed the slots freed by preempted $P_{\text{low}}$ pods.
* This satisfied the high-priority workload without needing further preemption.
* As a result, the remaining $P_{\text{low}}$ pods were never preempted and lacked `DeletionTimestamp`, failing the test assertion:
  ```text
  "Check all low priority pods to be about to preempted."
  ```

#### Resolution
PR #135372 enforced strict sequential staging:
1. Create all $P_{\text{med}}$ pods.
2. Wait until **all** $P_{\text{low}}$ pods have `DeletionTimestamp` populated.
3. Create all $P_{\text{high}}$ pods.
4. Verify that $P_{\text{high}}$ pods are prioritized and scheduled cleanly.

---

## 6. Architectural Patterns and Guidelines for Scheduler Test Suites

Based on the analysis of recent preemption test remediations, the following best practices should be maintained across Kubernetes scheduling test suites:

### 1. Granular Test Package Decomposition
* When a test suite introduces heavy multi-pod workloads (e.g., gang scheduling, composite pod groups, or large node topologies), isolate the suite into a dedicated subpackage (`test/integration/.../<feature>/`).
* Ensure subpackages instantiate their own `TestMain` and etcd storage harness to prevent cascading `KUBE_TIMEOUT` context cancellations.

### 2. Hierarchical Resource Reuse vs. Namespace Isolation
* Avoid instantiating new `kube-apiserver` instances for subtests that share identical static feature gate configurations.
* Use `testutils.WithNewNamespace` to provide clean, isolated namespaces for each subtest while reusing the parent API server and client pool.
* Explicitly clean up cluster-scoped resources (e.g., `Node` objects, cluster-wide `PriorityClass` resources) in `t.Cleanup()` blocks.

### 3. Namespace-Scoped Extended Resources
* Never hardcode cluster-wide resource names in parallel e2e tests.
* Always namespace dynamic resource types (e.g., `example.com/<namespace>`) to eliminate cross-test interference on shared nodes.

### 4. Deterministic State Synchronization
* Never rely on time-based sleeps (`time.Sleep`) or immediate unsynchronized property checks.
* When intercepting scheduler framework phases (`Permit`, `PreBind`, `PostFilter`), pass pre-populated control channels to plugin constructors before starting the scheduler.
* Explicitly poll pod condition transitions (e.g., `PodScheduled == True` or `DisruptionTarget` condition presence) using `wait.PollUntilContextTimeout`.

### 5. Calibrated Benchmark Workload Ratios
* When writing scheduler preemption benchmarks, clearly document and calibrate the victim-to-preemptor ratio (e.g., the 10:1 ratio in `PreemptionPodGroupDisruptionMode`).
* Standardize priority integer values across templates to prevent accidental preemption among test infrastructure pods.
* Provide dual-tier benchmark workloads: a fast, short integration workload for standard CI validation (`[integration-test, short]`) and a full-scale workload for performance evaluation (`[performance]`).

---

## 7. Chronological Reference & Commit Matrix

```text
2025-11-20  PR #135372 (commit cc5ec714c1f) - Fix priority ordering in async preemption e2e
2025-12-05  PR #135623 (commit eb137613381) - Enforce serial execution for RWOP volume preemption
2026-03-25  PR #138017 (commit 02d950fb520) - Fix race conditions and timing issues in preemption tests
2026-07-10  PR #140408 (commit 0ca3eac34b0) - Add topology spreading gang preemption perf scenarios
2026-07-16  PR #140651 (commit f8c84779032) - Add PodGroup victim disruption mode perf benchmarks
2026-07-20  PR #140737 (commit 1986d5df3e8) - Share API server across TestPreemption subtests
2026-07-21  PR #140737 (commit fd24803a947) - Reduce redundant feature gate overrides in TestPreemption
2026-07-23  PR #140872 (commit 362f57d8613) - Use namespaced extended resources in WAP e2e
2026-07-29  PR #141048 (commit 4dcbaf28403) - Split PodGroup preemption tests into own package
2026-09-09  PR #140408 (commit 9d21b8ec688) - Align priority values in gang preemption benchmark
```
