# Max Pod Count Exemption — Contract & Guardrails

This document captures the decisions made in Step 1 of the implementation plan.
Subsequent steps (helper, preFilter state, filter path, tests) MUST match this
contract.

## 1. Exemption mechanism

The exemption is expressed via a **pod annotation**, not a label.

- **Annotation key:** `kubelet.datadoghq.com/exclude-from-max-pods`
- **Truthy value:** the literal string `"true"` (case-sensitive).
- **Any other value, missing annotation, or malformed value:** treated as
  **not exempt** (fail-closed).

Rationale for annotation over label:
- Labels are selector-visible and routinely mutated by controllers / users for
  unrelated purposes; an accidental selector match could enable the exemption.
- Annotations are explicit metadata and are the conventional Kubernetes
  surface for opt-in behavior knobs (cf. `scheduler.alpha.kubernetes.io/...`,
  `kubernetes.io/...`).
- Existing kubelet/scheduler-level toggles (e.g. critical pod priority,
  PodOverhead opt-in via runtime class) follow the annotation pattern.

## 2. Scope of the exemption

Exemption applies **only to the pod-count check** — i.e. the
`v1.ResourcePods` / `allowedPodNumber` comparison performed inside
`noderesources.fitsRequest(...)`.

It does **not** affect:

- CPU requests/limits (`v1.ResourceCPU`)
- Memory requests/limits (`v1.ResourceMemory`)
- Ephemeral storage (`v1.ResourceEphemeralStorage`)
- Scalar / extended resources (GPUs, hugepages, custom device plugins)
- Any other scheduler plugin (taints, affinity, topology spread, volume
  limits, NodePorts, etc.)

An exempt pod whose CPU/memory request exceeds node capacity is still
rejected — same as any other pod.

## 3. Policy enforcement is external

The scheduler / kubelet code **does not** enforce *who* is allowed to set the
exemption annotation. It honours the annotation if present and truthy.

Policy (which namespaces, service accounts, or workloads may set this
annotation) is expected to live in:

- A **ValidatingAdmissionPolicy** / admission webhook in the cluster, OR
- An external mutating layer that strips the annotation from untrusted
  callers.

Code documentation added in step 2 MUST call this out explicitly so that an
operator reading the helper function understands the implicit trust model.

## 4. Hard restrictions to document in code

The helper's godoc (added in step 2) must state, at minimum:

1. The annotation key as a named constant.
2. That only the literal value `"true"` opts in; everything else is non-exempt.
3. That the exemption affects **only** `ResourcePods` accounting.
4. That **no in-tree enforcement** exists for which workloads may set the
   annotation — operators MUST gate this via admission.
5. That CNI / IPAM capacity, conntrack, and node-local resource pools are
   independent of the kubelet pod cap; bypassing the cap can saturate them.
   Operational alerting is required.

## 5. Non-goals

- Not a feature gate. The behavior is always on but only triggers when the
  annotation is present and truthy.
- Not a generic "resource exemption" framework. Only `ResourcePods` is in
  scope.
- Not a kubelet-only or scheduler-only change — both paths share the same
  decision via `noderesources.Fits(...)` / `AdmissionCheck(...)`.
- Not retroactive. Pods already running on a node are unaffected; the check
  only applies on the admission / scheduling decision for the *incoming* pod.

## 6. Symbol-level checklist for subsequent steps

Steps 2-5 will touch (per the plan):

- `pkg/scheduler/framework/plugins/noderesources/fit.go`
  - New constant: annotation key.
  - New helper: `isExcludedFromMaxPodCount(pod *v1.Pod) bool`.
  - `preFilterState` gains `ExcludeFromPodCount bool`.
  - `computePodResourceRequest` populates the new field.
  - `fitsRequest` signature gains `excludeFromPodCount bool` and uses `+0`
    instead of `+1` when true.
  - Callers updated: `Filter`, `Fits`, `isFit`.
- `pkg/scheduler/eventhandlers.go` / `pkg/kubelet/lifecycle/predicate.go`
  - Only signature plumbing if `Fits(...)` API changes; no separate logic.

Tests (steps 6-8) will live in:

- `pkg/scheduler/framework/plugins/noderesources/fit_test.go`
- `pkg/scheduler/eventhandlers_test.go`
- `pkg/kubelet/lifecycle/predicate_test.go`
