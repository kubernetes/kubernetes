# Step 1 handoff: Define behavior and guardrails

## What was done
- Authored `scratch/max-pod-count-exclusions/contract.md` capturing the
  exemption contract that steps 2-10 must implement against.
- Locked the exemption mechanism: pod **annotation**
  `kubelet.datadoghq.com/exclude-from-max-pods` with literal value `"true"`.
  Anything else (missing, empty, `"false"`, malformed) is non-exempt
  (fail-closed).
- Scoped the exemption to `v1.ResourcePods` only — CPU, memory, ephemeral
  storage, scalar/extended resources, and all other scheduler plugins are
  unaffected.
- Declared policy enforcement (which namespaces / service accounts may set
  the annotation) explicitly **out of scope** for in-tree code. It lives in
  an external admission policy / webhook.
- Listed the hard guardrails the step-2 helper godoc must state: the
  annotation constant, the strict `"true"` semantics, the `ResourcePods`
  scope, the lack of in-tree policy enforcement, and the operational
  warning about CNI / IPAM / conntrack capacity.
- Enumerated non-goals: not a feature gate, not a generic resource
  exemption framework, not kubelet-only or scheduler-only, not retroactive
  for already-running pods.
- Produced the symbol-level checklist that pins steps 2-5 to specific
  files and identifiers (`fit.go`, `preFilterState`,
  `computePodResourceRequest`, `fitsRequest`, `Filter`, `Fits`, `isFit`)
  and steps 6-8 to specific test files.
- Marked step 1 complete in `implementation-plan.md` via strikethrough +
  parenthetical pointer to the artifacts.

## Key decisions
- **Annotation over label**: labels are selector-visible and routinely
  mutated by controllers; an accidental selector match could enable the
  exemption. Annotations are the conventional opt-in surface for
  kubelet/scheduler behavior knobs.
- **Strict `"true"` only**: avoids ambiguity around `"1"`, `"yes"`,
  `"True"`. Anyone wanting the exemption must spell it exactly. This is
  fail-closed by design.
- **No in-tree policy enforcement**: keeps the kubelet/scheduler change
  minimal and reviewable. Restricting *who* may set the annotation is a
  cluster-config concern (ValidatingAdmissionPolicy / mutating webhook /
  Datadog-internal admission layer), not a scheduler concern.
- **Single source of truth in `noderesources`**: both scheduler filter
  and kubelet admission flow through `noderesources.Fits(...)` via
  `AdmissionCheck(...)`. No kubelet-only exemption logic.

## Issues encountered
- None. Step 1 is documentation-only; no tests were run because no code
  changed. The next step (helper in `fit.go`) is where compile + unit
  tests start mattering.

## Test state
- No tests were run. Step 1 produces only `contract.md`; there is no
  executable artifact to test. Test commands enumerated in step 10 will
  be exercised starting with step 2.

## Next step
- Step 2: add `isExcludedFromMaxPodCount(pod *v1.Pod) bool` helper plus
  the annotation-key constant in
  `pkg/scheduler/framework/plugins/noderesources/fit.go`, with godoc that
  enumerates the five guardrails listed in `contract.md` §4.
