# Step 8 handoff: Add kubelet predicate test for OutOfPods integration

## What was done
- Added `TestPredicateAdmitPodCountExemption` to
  `pkg/kubelet/lifecycle/predicate_test.go`.
- Test uses a single-pod-cap node (`makeResources`/`makeAllocatableResources`
  with pods=1) and one existing pod, so the pod-count dimension is the only
  failable check.
- Drives `predicateAdmitHandler.Admit(...)` directly via
  `NewPredicateAdmitHandler` with a noop `pluginResourceUpdateFunc`.
- Three sub-cases:
  - normal pod (no annotation) -> `Admit=false`, `Reason=OutOfPods`.
  - exempt pod (`kubelet.datadoghq.com/exclude-from-max-pods: "true"`) ->
    `Admit=true`.
  - malformed annotation value (`"True"`, wrong case) -> `Admit=false`,
    `Reason=OutOfPods` (locks in the fail-closed contract).
- Imports `noderesources` to reference
  `noderesources.ExcludeFromMaxPodCountAnnotationKey` so the test stays in
  sync with the single source of truth.

## Key decisions
- Reused existing `makeResources`/`makeAllocatableResources` helpers from
  `predicate_test.go` rather than building a node literal by hand, matching
  the style of neighbouring tests.
- Validated only `Admit` and `Reason` (not `Message`), keeping the test
  resistant to message-string churn while still exercising the
  `v1.ResourcePods -> OutOfPods` reason mapping unique to the kubelet path.
- Did not duplicate scheduler-level scenarios (non-pod resource still
  enforced for exempt pod, direct `Fits` paths) — those are already
  covered in `fit_test.go` and `eventhandlers_test.go` per steps 6 and 7.

## Test state
- `go test ./pkg/kubelet/lifecycle/... -run TestPredicateAdmitPodCountExemption`
  passes for all three sub-cases.
- Full `pkg/kubelet/lifecycle` test suite remains green.

## Next step
- Step 9: add policy and rollout artifacts (admission policy + short
  design/readme note covering CNI/IPAM impact and observability). Note
  the plan flags this as primarily cluster-config-repo work, not
  kubernetes-repo work, so the next iteration should decide whether the
  in-repo portion is just documentation.
