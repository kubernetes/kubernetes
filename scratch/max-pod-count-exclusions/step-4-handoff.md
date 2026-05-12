# Step 4 handoff: Teach scheduler filter path to skip pod-count increment for exempt incoming pod

## What was done
- Modified `pkg/scheduler/framework/plugins/noderesources/fit.go`:
  - In `fitsRequest(...)`, introduced a local `podCountIncrement` variable that defaults to `1` and is set to `0` when `podRequest.ExcludeFromPodCount` is true (the flag set in step 3 from the exemption annotation).
  - Replaced the `len(nodeInfo.Pods)+1 > allowedPodNumber` check with `len(nodeInfo.Pods)+podCountIncrement > allowedPodNumber`, so exempt pods are no longer blocked by `Allocatable.Pods`.
  - Updated the `Requested` field on the emitted `InsufficientResource` to `int64(podCountIncrement)` so reported usage matches what the pod actually contributes.
  - Added a doc comment explaining the contract, that only the pod-count dimension is affected, and pointing readers at `isExcludedFromMaxPodCount` for the full policy.
- The exemption value is read off `preFilterState.ExcludeFromPodCount`, which already flows through both `Filter` (via `getPreFilterState`) and the direct `Fits(...)` path (via `computePodResourceRequest`) from steps 2 and 3. No call-site signature changes were required to thread the boolean — it rides on `preFilterState` — so `Filter`, `Fits`, and `isFit` were left untouched in this step.

## Key decisions
- Threaded the exemption via `preFilterState.ExcludeFromPodCount` rather than adding a new parameter to `fitsRequest`. The plan suggested an extra boolean parameter, but the preFilter state already reaches every caller (`Filter`, `Fits`, `isFit`) and steps 2/3 explicitly placed the flag there to centralize the decision. Adding a parameter would be redundant and would force all callers to recompute or pass through a field that is already on the struct they pass in.
- Kept `Requested` on the `InsufficientResource` honest (`int64(podCountIncrement)`) instead of hard-coding `1`. For exempt pods this branch is unreachable when `len(Pods) <= allowedPodNumber`, but if a node is already over capacity for unrelated reasons the reported "Requested" stays consistent with the actual increment.

## Test state
- No tests run in this step. Step 6 of the plan covers scheduler unit tests for this behavior (`fit_test.go`), step 7 covers admission-check tests, step 8 covers kubelet predicate tests, and step 10 covers the full validation pass.

## Next step
- Step 5: Keep kubelet admission aligned through shared `AdmissionCheck` path — verify `scheduler.AdmissionCheck(...)` still routes to `noderesources.Fits(...)` and benefits from the exemption decision without kubelet-only logic.
