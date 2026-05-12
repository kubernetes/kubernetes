# Step 7 handoff: Add admission-check tests to verify kubelet-visible behavior

## What was done
- Extended `TestAdmissionCheck` in `pkg/scheduler/eventhandlers_test.go`
  with three new subcases that exercise the pod-count exemption through
  the `scheduler.AdmissionCheck(...)` seam (the same path the kubelet
  takes via `pkg/kubelet/lifecycle/predicate.go`).
- Added a shared `tooManyPodsError` AdmissionResult fixture and an
  `onePod` capacity map that pins the node's `ResourcePods` capacity to
  1 so a single existing pod saturates the pod-count dimension.
- New subcases:
  - `pod-count exemption: normal pod rejected when node is at pod cap`
    -- baseline; asserts the standard `Too many pods`
    InsufficientResource entry is returned for a non-exempt pod.
  - `pod-count exemption: exempt pod admitted when node is at pod cap`
    -- asserts an empty admission-result slice (`nil`) when the pod
    carries the exemption annotation with value `"true"`.
  - `pod-count exemption: malformed annotation value is non-exempt`
    -- fail-closed guardrail; asserts that annotation value `"True"`
    (wrong case) still yields the pod-count rejection, mirroring the
    fit_test.go helper-level coverage but through AdmissionCheck.
- All three subcases pass under `go test ./pkg/scheduler/ -run
  TestAdmissionCheck`.

## Key decisions
- Used `wantAdmissionResults: [][]AdmissionResult{nil, nil}` for the
  exempt admit case (rather than an empty `{}`) to match how the
  existing test loop initialises an empty result slice.
- Annotation key is referenced via the exported
  `noderesources.ExcludeFromMaxPodCountAnnotationKey` constant so the
  test cannot silently drift from the implementation.
- Each new subcase has a focused comment block explaining what the
  case pins down, since the matrix is long and the intent of the new
  rows isn't obvious from the table alone.

## Test state
- `go test ./pkg/scheduler/ -run TestAdmissionCheck` -- PASS
  (all original cases plus the three new pod-count-exemption cases).
- No tests were skipped.

## Next step
- Step 8: Add kubelet predicate test for OutOfPods integration in
  `pkg/kubelet/lifecycle/predicate_test.go`, covering normal vs
  exempt pods when the node is at the pod cap.
