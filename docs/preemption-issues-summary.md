# Summary and Taxonomy of Recent Kubernetes Pod Preemption Issues

## Executive Summary

Over the recent development cycles (late 2025 through late 2026), the Kubernetes preemption subsystem has undergone substantial expansion, architectural refactoring, bug fixing, and performance optimization. The introduction of major features—notably **Workload-Aware Preemption (WAP) / Gang Scheduling (KEP-5710)**, **CompositePodGroups (KEP-6012)**, **In-Place Pod Vertical Scaling Preemption (KEP-1287)**, and **Asynchronous Preemption**—introduced complex state machine interactions across the scheduler framework, API server, and node runtime.

This document synthesizes and categorizes the recent issues, fixes, and improvements across five distinct operational domains and defines the task decomposition for deep-dive investigation.

---

## 1. Categorization of Recent Issues

### Domain 1: Workload-Aware Preemption (WAP) & PodGroup Gang Scheduling
Addresses the preemption of multiple pods as cohesive workloads/gangs, CompositePodGroup hierarchies, preemption policy propagation, and priority authoritative calculations.

* **PR #141932 (`brejman/pg-preemption-extensions`)**: Added `PreemptionExtensions` to `PodGroupPostFilter` to allow custom extension hooks during gang preemption evaluations.
* **PR #141934 (`macsko/refactor_pod_group_preemption`)**: Refactored pod group preemption logic to operate against the `GenericPodGroup` abstraction.
* **PR #141930 (`macsko/validate_child_pod_groups_for_equal_priority`)**: Enforced validation that child pod groups share equal priority and `preemptionPolicy` across the hierarchy.
* **PR #140634 (`tosi3k/cpg-wap`)**: Added support for CompositePodGroups in workload-aware preemption, adding `DisruptionMode` and `PreemptionPolicy` fields to the API.
* **PR #140871 (`Argh4k/fix-podgroupcycle`)**: Fixed state pollution by clearing pod group cycle state during pod group preemption.
* **PR #140745 (`tosi3k/wap-snapshot`)**: Updated `PodEligibleToPreemptOthers` to use the `PodGroup` snapshot lister instead of raw cache lookups.
* **PR #140641 / PR #140590 / PR #138967 / PR #139280**: Supported and propagated `NominatedNodeName` (NNN) across all pods in a pod group; resolved early exit race conditions during ongoing preemption.
* **PR #140311 & PR #140180**: Propagated preemption status messages to `PodGroupStatus` and surfaced user-visible scheduling failure/success annotations.
* **PR #140359, PR #140312, PR #139240**: Enforced matching `PreemptionPolicy` across all pods in a gang and added `PreemptionPolicy` to `PodGroupTemplate` and `PodGroup`.
* **PR #139520**: Merged legacy `GangScheduling` and `WorkloadAwarePreemption` feature gates under `GenericWorkload`.
* **PR #139030**: Ensured `PodGroup` priority is strictly authoritative during workload preemption decisions.
* **PR #138710**: Matched preemptor eligibility behavior between single pod and pod group preemption flows.
* **PR #138886 & PR #138757**: Corrected reprieval logic to ensure `maxScheduledCount` cannot decrease during the victim reprieval process for gang workloads.

---

### Domain 2: Default Preemption Algorithm, Victim Ordering, PDBs & Extenders
Addresses the core in-tree `DefaultPreemption` plugin, victim selection determinism, PodDisruptionBudget (PDB) parsing, and scheduler extender integration.

* **PR #135486 (commit `cb33cf457d0`)**: Fixed preemption failure when using filter extenders. Prevented dropping empty-victim nodes prematurely before preempt-capable extenders could evaluate and nominate victims.
* **PR #141785 (commit `82dead7c815`)**: Fixed scheduler preemption ignoring empty PDB selectors (empty selectors select all pods in the namespace).
* **PR #140999 (commit `24127a76250` / `8ffd4531cb6`)**: Made `MoreImportantVictim` ordering deterministic when victim pods have identical start times or unstarted timestamps.
* **PR #140054 (commit `bd6cae4a3fe`, `de33b423915`, `8d5731c016d`)**: Addressed scheduler flakes by properly reactivating preemptor pods after in-memory preemption cycles and standardizing in-memory preemption results.
* **PR #136613**: Decoupled victim evaluation from victim execution in the preemption framework.
* **PR #134927**: Prevented attempting preemption against pods that already have `DeletionTimestamp` set.

---

### Domain 3: Asynchronous Preemption, Scheduling Queue Flushes & Preemptor Starvation
Addresses queue management, preventing preemptor pods from getting permanently stuck in the unschedulable queue, permit handling, and prebind phase cancellations.

* **PR #139162, PR #139330, PR #139331 (`brejman/fix-stuck-preemption*`)**: Resolved bug where preemptor pods got permanently stuck in the unschedulable queue. Ensured gated pods flush at equal frequencies and properly reset `WasFlushedFromUnschedulable`.
* **PR #135502 (`Argh4k/binding-pods`)**: Implemented prebind-phase preemption without API server deletion calls by canceling the prebind context and requeuing to the backoff queue.
* **PR #135719 (`Argh4k/waiting-pod-integration-test`)**: Ensured pods preempted while waiting on permit (`WaitOnPermit`) are routed to the backoff queue rather than discarded.
* **PR #135495 (`tosi3k/skip-last-pod-deletion`)**: In async preemption, skip evicting subsequent victims if any prior victim deletion returns an error.
* **PR #134730 (`ania-borowiec/verify_ongoing_preemption`)**: Added checks for ongoing async preemption before evicting pods, preventing race conditions when higher-priority preemptors arrive.
* **PR #134294 (`ania-borowiec/test_for_rollback`)**: Fixed premature reactivation of preemptor pods while victim deletion is still pending in the API server.
* **PR #135955 & PR #139373**: Aligned victim metric semantics between async and sync preemption modes.

---

### Domain 4: In-Place Pod Vertical Scaling (IPPVS) Preemption (KEP-1287)
Addresses scheduler-coordinated resource preemption to enable in-place pod resizing without pod restarts.

* **PR #140000 (`natasha41575/ippr-preemption`)**: Introduced scheduler preemption for in-place pod resize under feature gate `InPlacePodVerticalScalingSchedulerPreemption`.
* **Plugin Implementation (`2fa5a2eda68`)**: Added preemption plugin to handle deferred resize pods upon node preemption policy updates.
* **Kubelet Bypass (`36e85e715eb`)**: Allowed Kubelet to bypass local admission preemption when handling resize requests initiated by scheduler.
* **APIServer & Validation (`9f663aa01a8`, `efb871ca6bb`)**: Added schema validation and feature gate controls for resize-induced preemption.
* **Metrics (`PR #140122`)**: Added latency and queue duration metrics for deferred resize preemption operations.

---

### Domain 5: Test Suite Stability, Flakes, Benchmarking & Modularization
Addresses runtime exhaustion in integration test packages, deadlock/race conditions, and gang preemption performance test suites.

* **PR #141048 (`BenTheElder/preemption-split-podgroup`)**: Split `PodGroup` preemption integration tests into a dedicated package (`test/integration/scheduler/preemption/podgroup`) to prevent exceeding the shared 600s `KUBE_TIMEOUT`.
* **PR #140737 (`shwetha-s-poojary/fix_flake_TestPreemption`)**: Shared single API server instances across `TestPreemption` subtests, reducing startups from 72 to 8 and cutting test duration from ~360s to ~70s.
* **PR #140408 & PR #140651**: Introduced topology spreading gang preemption benchmark scenarios and pod-preempts-podgroup benchmarks.
* **PR #140872 & PR #140602**: Fixed test flakes in WAP related to per-namespace extended resource allocations.
* **PR #138017**: Eliminated race conditions and timing flakes by synchronizing node assignment with mutex protection in preemption tests.

---

## 2. Decomposed Task Execution Plan

To produce comprehensive, deep-dive technical summaries of each area before final consolidation, the following task hierarchy is filed:

1. **Task 1 (WAP & Gang Preemption)**: Deep dive into KEP-5710 / KEP-6012, `PodGroupPostFilter`, `GenericPodGroup`, and victim reprieval algorithms.
2. **Task 2 (Default Preemption & Victim Ordering)**: Deep dive into victim ordering determinism, PDB matching logic, and extender interactions.
3. **Task 3 (Async Preemption & Queue Invariants)**: Deep dive into unschedulable queue starvation, prebind context cancellation, and permit waiting.
4. **Task 4 (In-Place Resize Preemption)**: Deep dive into KEP-1287 scheduler preemption for pod resize and kubelet interactions.
5. **Task 5 (Test Infrastructure & Performance)**: Deep dive into integration test partitioning, API server lifecycle, and benchmark metrics.
6. **Task 6 (Final Consolidation)**: Final synthesis of all 5 domain summaries into the definitive preemption architecture and troubleshooting guide.
