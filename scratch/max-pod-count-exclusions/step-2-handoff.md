# Step 2 handoff: Add shared pod-exemption helper in node resources plugin

## What was done
- Edited `pkg/scheduler/framework/plugins/noderesources/fit.go` to introduce
  the shared pod-exemption surface that later steps will consume:
  - Exported constant `ExcludeFromMaxPodCountAnnotationKey =
    "kubelet.datadoghq.com/exclude-from-max-pods"` placed in the existing
    `const` block alongside `preFilterStateKey` / `preScoreStateKey`.
  - Private constant `excludeFromMaxPodCountAnnotationValue = "true"` --
    kept unexported so callers cannot accidentally accept other
    truthy-looking spellings; they must go through the helper.
  - Helper `isExcludedFromMaxPodCount(pod *v1.Pod) bool` placed immediately
    above `isFit(...)`. Returns `false` for a nil pod, otherwise compares
    `pod.Annotations[ExcludeFromMaxPodCountAnnotationKey]` against the
    literal `"true"`.
- Authored the godoc on `isExcludedFromMaxPodCount` to enumerate the five
  guardrails from `contract.md` §4: annotation-not-label surface, strict
  `"true"` semantics, scope limited to `v1.ResourcePods`, no in-tree
  policy enforcement, and the CNI/IPAM/conntrack operational warning.
  Godoc on `ExcludeFromMaxPodCountAnnotationKey` also calls out the
  fail-closed semantics and points back to the helper.
- No call sites were modified -- the helper is staged for steps 3 and 4 to
  wire into `preFilterState` / `computePodResourceRequest` / `fitsRequest`.
  This keeps step 2 minimal and reviewable in isolation.

## Key decisions
- **Unexported value constant**: `excludeFromMaxPodCountAnnotationValue`
  stays private so the only way to ask "is this pod exempt?" is through
  `isExcludedFromMaxPodCount`. Prevents accidental drift to `"1"` /
  `"yes"` / `"True"` semantics across future callers.
- **Nil-pod guard returns `false`**: matches the fail-closed contract.
  No caller in scheduler or kubelet should ever pass nil, but defending
  against it keeps the helper safe to call from any future hook point
  (e.g. score extensions, debug logging) without panic risk.
- **Helper kept private (`isExcludedFromMaxPodCount`)**: the *annotation
  key* is exported because external admission policies and tests may
  need the string, but the decision function stays package-private --
  every consumer in the tree lives in `pkg/scheduler/framework/plugins/
  noderesources` or routes through `noderesources.Fits(...)` /
  `scheduler.AdmissionCheck(...)`. Avoids prematurely committing to a
  public API surface.
- **Placement near `isFit`**: groups the new helper with the other
  per-pod fit-time predicates rather than near the top-of-file
  pre-filter scaffolding. Step 3 will reach for it from
  `computePodResourceRequest`, which is the next function above.

## Test state
- `go build ./pkg/scheduler/framework/plugins/noderesources/...` --
  passes (no output). Confirms the new constants and helper compile
  cleanly into the existing package.
- No unit tests were added or run in this step. The helper is currently
  dead code (no callers), so a direct unit test would only assert the
  trivial equality. Step 6 will exercise it indirectly via `Filter` /
  `Fits` table tests once steps 3-4 wire it in; that is the right place
  to cover normal pod / exempt pod / malformed-annotation cases.
- Wider test suites (`go test ./pkg/scheduler/...`,
  `go test ./pkg/kubelet/lifecycle/...`) were deferred to step 10 per
  the plan; running them here would be noise with no call sites yet.

## Next step
- Step 3: extend `preFilterState` in the same `fit.go` with an
  `ExcludeFromPodCount bool` field, set it inside
  `computePodResourceRequest(...)` by calling
  `isExcludedFromMaxPodCount(pod)`, and leave all other resource-request
  computation untouched.
