# Step 5 handoff: Keep kubelet admission aligned through shared AdmissionCheck path

## What was done
- Verified that `scheduler.AdmissionCheck(...)` in `pkg/scheduler/eventhandlers.go` still
  routes to `noderesources.Fits(...)` unchanged. Because step 4 plumbed the
  exemption through `preFilterState` / `computePodResourceRequest` / `fitsRequest`,
  the kubelet admission path automatically picks up the same exemption decision
  without any signature change.
- Added a doc comment to `AdmissionCheck` in `pkg/scheduler/eventhandlers.go`
  explaining that the pod-count exemption is honoured transparently via
  `noderesources.Fits`, and that there is intentionally no kubelet-only branch
  (single source of truth lives in `noderesources`).
- Confirmed `pkg/kubelet/lifecycle/predicate.go` needs no changes: it calls
  `scheduler.AdmissionCheck(...)` and inherits the exemption behaviour.

## Key decisions
- Did not introduce any new symbols, parameters, or kubelet-side logic. The
  contract from step 1 explicitly requires a single source of truth in
  `noderesources`; any kubelet-only check would risk drift.
- Documented the exemption flow at the `AdmissionCheck` doc comment rather than
  at the call site in kubelet, since `AdmissionCheck` is the shared seam.

## Test state
- No test runs were required for this step: it is doc-only and the behavioural
  validation belongs to steps 6-8 (scheduler unit tests, admission-check tests,
  and kubelet predicate tests).

## Next step
- Step 6: add scheduler unit tests for pod-count exemption behaviour in
  `pkg/scheduler/framework/plugins/noderesources/fit_test.go` (normal pod
  blocked at cap, exempt pod admitted at cap, non-pod resources still enforced
  for exempt pod, malformed annotation treated as non-exempt; cover both
  `Filter` and direct `Fits(...)` paths).
