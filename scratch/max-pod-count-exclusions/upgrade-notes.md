# Max Pod Count Exemption — Upgrade & Rebase Notes

Companion to `contract.md` (in-tree behavior) and `design-note.md`
(operational contract). This document lists every in-tree symbol the
Datadog patch touches so that an engineer rebasing against upstream
Kubernetes can find, audit, and reapply the change without re-deriving
it from a diff.

Use this as the checklist when a kubernetes-repo rebase fails on any of
the files below. If a touched symbol's surrounding code has been
materially refactored upstream, treat that as a rebase-blocking event
and re-read `contract.md` §6 before proceeding — the contract pins what
the patch must preserve regardless of upstream churn.

## 1. Source files touched

| File                                                            | Why touched                                                                                                   |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `pkg/scheduler/framework/plugins/noderesources/fit.go`          | Annotation constant, exemption helper, `preFilterState` field, `computePodResourceRequest`, `fitsRequest`, callers. |
| `pkg/scheduler/framework/plugins/noderesources/fit_test.go`     | Table-driven coverage for normal / exempt / malformed-annotation pods on `Filter` and `Fits`.                 |
| `pkg/scheduler/eventhandlers.go`                                | Signature plumbing only (`AdmissionCheck` routes through `Fits`).                                             |
| `pkg/scheduler/eventhandlers_test.go`                           | `TestAdmissionCheck` cases covering exempt vs non-exempt admission via the kubelet-visible entry point.       |
| `pkg/kubelet/lifecycle/predicate_test.go`                       | `TestPredicateAdmitPodCountExemption` covering `OutOfPods` reason mapping.                                    |
| `pkg/scheduler/metrics/metrics.go`                              | Temporary `PodCountExemptionAdmissionsTotal` counter, registered in `metricsList`.                            |
| `scratch/max-pod-count-exclusions/`                             | Contract, design note, reference admission policy, this upgrade note, step handoffs.                          |

`pkg/kubelet/lifecycle/predicate.go` is **deliberately not** touched —
the kubelet path consumes the exemption transitively through
`scheduler.AdmissionCheck` → `noderesources.Fits`. Any local logic
added to `predicate.go` would violate the single-source-of-truth
property documented in `contract.md` §5.

## 2. Exported / package-level symbols

These are the symbols a rebase must preserve. Each row links a symbol
to the contract clause that motivates it; if upstream renames or moves
one of these, the rebase needs to follow the symbol, not the line
number.

| Symbol                                                                          | Where                                                              | Contract clause                                                                                  |
| ------------------------------------------------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| `ExcludeFromMaxPodCountAnnotationKey` (exported `const string`)                 | `pkg/scheduler/framework/plugins/noderesources/fit.go`             | `contract.md` §1 (annotation key is part of the public surface; admission policy matches on it). |
| `excludeFromMaxPodCountAnnotationValue` (private `const string = "true"`)       | same file                                                          | `contract.md` §1 (only the literal `"true"` opts in — fail-closed).                              |
| `isExcludedFromMaxPodCount(pod *v1.Pod) bool` (private helper)                  | same file                                                          | `contract.md` §3, §4 (single decision function; godoc enumerates the five guardrails).           |
| `preFilterState.ExcludeFromPodCount` (private field, bool)                      | same file                                                          | `contract.md` §6 (resolved once at PreFilter time, observed by every Filter call).               |
| `computePodResourceRequest(pod, opts)` (populates the new field)                | same file                                                          | `contract.md` §6 (sets `ExcludeFromPodCount` from the helper).                                   |
| `fitsRequest(...)` signature (gains `excludeFromPodCount bool`)                 | same file                                                          | `contract.md` §6 (`+0` instead of `+1` against `allowedPodNumber` when true).                    |
| `Fits(...)`, `isFit(...)`, `Filter(...)` (callers of `fitsRequest`)             | same file                                                          | `contract.md` §6 (every entry point must pass the boolean through).                              |
| `AdmissionCheck(...)`                                                           | `pkg/scheduler/eventhandlers.go`                                   | `contract.md` §5 (kubelet reuses the scheduler decision via this function).                      |
| `PodCountExemptionAdmissionsTotal` (`*metrics.CounterVec`, temporary)           | `pkg/scheduler/metrics/metrics.go`                                 | `design-note.md` §3b (rollout observability; may be removed once the rollout has soaked).        |

## 3. Rebase decision tree

1. **Upstream renamed `fitsRequest`** → follow the rename; keep the
   `excludeFromPodCount` parameter in the same logical position
   (currently last). Update all in-tree callers.
2. **Upstream changed the `preFilterState` shape** (e.g. moved
   `Resource` into a sub-struct) → keep `ExcludeFromPodCount` at the
   top level of `preFilterState` so existing tests that construct the
   struct by literal continue to compile. If upstream introduces its
   own per-pod boolean state, do **not** fold this in — the field
   carries a Datadog-specific contract.
3. **Upstream changed `computePodResourceRequest` signature** (e.g.
   the `TODO(ndixita)` to take `ResourceRequestOptions` finally lands)
   → keep the post-`SetMaxResource` block that sets `ExcludeFromPodCount`
   and emits the V(4) log + counter. Order matters: the field must be
   set on the same `*preFilterState` that the caller writes to
   `CycleState`.
4. **Upstream changed `AdmissionCheck`'s call to `Fits`** → re-derive
   how the boolean is plumbed but **do not** add a parallel path in
   `pkg/kubelet/lifecycle`. The kubelet must keep consuming the
   scheduler decision; a second decision site is a contract violation
   per `contract.md` §5.
5. **Upstream removed the V(4) log or the counter** (cherry-pick
   conflict) → either is replaceable; restore both. The counter is
   marked temporary in its godoc and may be retired by an
   intentional in-repo PR after a soak, but a rebase that removes it
   silently is a regression.
6. **Tests fail after a rebase with an obvious behavioural diff** →
   re-read the relevant scenario in `contract.md` §2 (scope) or §4
   (hard restrictions). The tests pin the contract; if a behavioural
   diff is acceptable, the contract changes first, then the tests.

## 4. What does NOT need to be rebased

- Files under `scratch/max-pod-count-exclusions/` are documentation
  and reference artifacts; they live outside upstream's tree and are
  only relevant inside this fork.
- `admission-policy.yaml` is a reference shape for the cluster-config
  repo. It is **not** intended to be applied by this repo's CI and
  has no upstream counterpart.

## 5. Touched-symbol summary (TL;DR for reviewers)

> `Fits`, `fitsRequest`, `preFilterState`,
> `ExcludeFromMaxPodCountAnnotationKey`,
> `excludeFromMaxPodCountAnnotationValue`,
> `isExcludedFromMaxPodCount`,
> `computePodResourceRequest`,
> `AdmissionCheck`,
> `PodCountExemptionAdmissionsTotal`.

If a rebase preserves all of the above and the three targeted test
suites (`pkg/scheduler/framework/plugins/noderesources/...`,
`pkg/scheduler/...`, `pkg/kubelet/lifecycle/...`) pass, the rebase is
complete.
