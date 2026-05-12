# Max Pod Count Exclusions Implementation Plan

1. ~~Define behavior and guardrails~~ **(done — see `contract.md` and `step-1-handoff.md`)**
   - Decide the exemption contract: use an annotation (recommended) such as `kubelet.datadoghq.com/exclude-from-max-pods: "true"` instead of an arbitrary label.
   - Define hard restrictions in code docs: only specific namespaces or service accounts are allowed to use the exemption.
   - Add clear non-goals: exemption affects only pod-count (`ResourcePods`), not CPU/memory/storage/resource requests.

2. ~~Add shared pod-exemption helper in node resources plugin~~ **(done — see `step-2-handoff.md`)**
   - File: `pkg/scheduler/framework/plugins/noderesources/fit.go`
   - Add a helper function (for example `isExcludedFromMaxPodCount(pod *v1.Pod) bool`) near existing fit logic.
   - Initially implement annotation check and keep logic centralized so scheduler and kubelet paths consume the same decision path.
   - Add comments documenting security expectations and that policy enforcement is external (admission).

3. ~~Extend preFilter state to carry pod-count exemption~~ **(done — see `step-3-handoff.md`)**
   - File: `pkg/scheduler/framework/plugins/noderesources/fit.go`
   - Extend `preFilterState` with a boolean (for example `ExcludeFromPodCount bool`).
   - Update `computePodResourceRequest(...)` to compute and set this boolean based on the incoming pod.
   - Keep resource request behavior unchanged for all other resources.

4. ~~Teach scheduler filter path to skip pod-count increment for exempt incoming pod~~ **(done — see `step-4-handoff.md`)**
   - File: `pkg/scheduler/framework/plugins/noderesources/fit.go`
   - Update `fitsRequest(...)` signature to accept the exemption boolean (for example `excludeFromPodCount bool`).
   - Change `allowedPodNumber` check:
     - today: `len(nodeInfo.Pods)+1 > allowedPodNumber`
     - target: use `+0` for exempt incoming pods and `+1` for normal pods.
   - Update all internal callers (`Filter`, `Fits`, and `isFit`) to pass the right value.

5. ~~Keep kubelet admission aligned through shared AdmissionCheck path~~ **(done — see `step-5-handoff.md`)**
   - Files: `pkg/scheduler/eventhandlers.go`, `pkg/kubelet/lifecycle/predicate.go` (minimal or no direct logic change expected)
   - Validate that `scheduler.AdmissionCheck(...)` still routes to `noderesources.Fits(...)` and now benefits from the same exemption decision.
   - Only adjust function calls/signatures in `AdmissionCheck(...)` if needed due to `Fits(...)` API changes.
   - Do not add separate kubelet-only exemption logic; keep single-source-of-truth in `noderesources`.

6. ~~Add scheduler unit tests for pod-count exemption behavior~~ **(done — see `step-6-handoff.md`)**
   - File: `pkg/scheduler/framework/plugins/noderesources/fit_test.go`
   - Add table-driven tests for:
     - normal pod blocked at max pod count
     - exempt pod admitted at max pod count
     - non-pod resources still enforced for exempt pod
     - malformed annotation values treated as non-exempt
   - Ensure tests cover `Filter` and direct `Fits(...)` paths.

7. ~~Add admission-check tests to verify kubelet-visible behavior~~ **(done — see `step-7-handoff.md`)**
   - File: `pkg/scheduler/eventhandlers_test.go`
   - Add/extend `TestAdmissionCheck` cases that show:
     - `AdmissionCheck(...)` returns pod-limit failure for normal pod
     - `AdmissionCheck(...)` does not return pod-limit failure for exempt pod
   - This validates kubelet behavior indirectly because kubelet uses `AdmissionCheck(...)`.

8. ~~Add kubelet predicate test for OutOfPods integration~~ **(done — see `step-8-handoff.md`)**
   - File: `pkg/kubelet/lifecycle/predicate_test.go`
   - Add tests where node is at pod cap:
     - normal pod yields `OutOfPods`
     - exempt pod is admitted (or at least not rejected for pod-count)
   - Keep coverage focused on admission result mapping and avoid duplicating scheduler tests.

9. Add policy and rollout artifacts (recommended for production)
   - Add an admission policy (in your cluster config repo, not necessarily this code repo) to:
     - enforce immutable exemption annotation
     - restrict usage to trusted namespaces/service accounts
   - Document operational safeguards in a short design/readme note:
     - expected impact on CNI/IPAM capacity
     - required observability and alerting for node pod density.

10. Validate, gate rollout, and prepare upgrade guidance
   - Run targeted tests:
     - `go test ./pkg/scheduler/framework/plugins/noderesources/...`
     - `go test ./pkg/scheduler/...`
     - `go test ./pkg/kubelet/lifecycle/...`
   - Add a temporary metric/log for exempt pod admissions to monitor usage during rollout.
   - Write an upgrade note listing touched symbols (`Fits`, `fitsRequest`, `preFilterState`) so rebases against upstream are straightforward.

