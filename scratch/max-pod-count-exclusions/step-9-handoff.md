# Step 9 handoff: Add policy and rollout artifacts

## What was done
- Added `scratch/max-pod-count-exclusions/design-note.md` — companion to
  `contract.md` covering the operational contract that must be satisfied
  outside this repo before the exemption is used in production. Sections:
  1. Admission policy expectations (ValidatingAdmissionPolicy preferred,
     mutating webhook as fallback, fail-closed `Fail` mode, explicit
     non-goals such as no in-tree allow-list and no feature gate).
  2. Expected impact on CNI/IPAM/node-local capacity, including the
     table of independent per-node limits (IPAM, conntrack, ARP, FDs,
     PIDs, cgroups, kube-proxy) and a four-item pre-flight checklist.
  3. Required observability and alerting (per-node metrics, exemption
     usage visibility via audit log scrape or admission webhook metric,
     five minimum alerts, Datadog-specific notes).
  4. Five-phase staged rollout plan with empty-allow-list staging gate
     and ≥ 7 day soak before broad rollout.
  5. Cross-link from in-tree symbols to this design note so reviewers
     coming from either side land in the same place.
- Added `scratch/max-pod-count-exclusions/admission-policy.yaml` —
  reference `ValidatingAdmissionPolicy` + parameter `ConfigMap` +
  `ValidatingAdmissionPolicyBinding` enforcing:
  1. Annotation immutability post-creation (UPDATE rejected if value
     changes).
  2. Namespace allow-list (empty default = deny-all on land).
  3. Service-account allow-list on CREATE (UPDATE skips SA re-check so
     unrelated status updates by other SAs are not rejected for merely
     carrying the unchanged annotation).
  4. Literal-value guardrail: only `"true"` accepted, anything else
     rejected with `reason: Invalid` so operators do not ship pods that
     look exempt but are not (per the fail-closed contract).
  - Ships with `failurePolicy: Fail` and `validationActions: [Deny,
    Audit]` so the audit trail captures every exemption-related
    decision.
  - Cross-linked to the in-tree contract via annotations on the
    `ValidatingAdmissionPolicy` so reviewers chasing from either side
    find the matching artifact.
- No Go code changed. No tests added or modified.

## Key decisions
- Kept both artifacts under `scratch/max-pod-count-exclusions/` rather
  than promoting them into the kubernetes-repo tree. Rationale: the
  plan explicitly notes the admission policy lives "in your cluster
  config repo, not necessarily this code repo"; shipping these files
  alongside `contract.md` keeps the in-tree contract, the operational
  contract, and the reference policy co-located for reviewers without
  implying they will be applied by this repo's CI.
- Chose `ValidatingAdmissionPolicy` (CEL, Kubernetes 1.30+ GA) over a
  webhook for the reference shape. Trade-off documented in
  `design-note.md` §1b: a mutating webhook that strips the annotation
  is offered as the fallback for older clusters because it degrades to
  a no-op (pod admitted but cap still applies) rather than a hard
  reject, which is easier to roll out but harder to audit.
- Param `ConfigMap` ships with both allow-lists empty so the policy
  lands in deny-all mode. Adding namespaces / SAs requires an explicit
  reviewable PR, matching the rollout plan's "land empty, opt in per
  namespace" gating.
- Service-account check applies only on CREATE, not UPDATE. Otherwise
  a kubelet status update from a different SA than the original
  creator would be rejected just for carrying the unchanged
  annotation. Immutability (validation #1) covers the UPDATE path's
  privilege-escalation surface; SA gating only needs to cover the
  CREATE path.
- Used `params.data.<key>.split(',')` with `.map(s, s.trim())` for
  allow-list parsing rather than a structured list type so operators
  can edit the `ConfigMap` from any tool (kubectl edit, helm values,
  argocd diff) without re-shaping the data.

## Issues encountered
- None. Step is documentation-only; no code paths touched, no tests
  run, no compilation involved.
- The two new files are caught by the user's `~/.gitignore_global`
  pattern `scratch/`, same as the earlier handoffs in this plan.
  Resolved by force-adding (`git add -f`) consistent with how
  `step-1-handoff.md` through `step-8-handoff.md` were committed.

## Test state
- No tests run. Step 9 produces documentation and a reference policy
  YAML; no Go packages were modified. Per the plan, the targeted test
  re-run is part of step 10 ("Validate, gate rollout, and prepare
  upgrade guidance"), not step 9.

## Next step
- Step 10: Validate and prepare upgrade guidance. Run the three
  targeted suites listed in the plan (`pkg/scheduler/framework/plugins/noderesources/...`,
  `pkg/scheduler/...`, `pkg/kubelet/lifecycle/...`), add a
  temporary metric/log for exempt pod admissions (note: the design
  note explicitly states the in-tree code does NOT emit such a
  metric — step 10 must decide whether to revisit that decision or
  to leave the observability burden on the cluster-config side as
  documented), and write the upgrade note listing the touched
  symbols (`Fits`, `fitsRequest`, `preFilterState`,
  `ExcludeFromMaxPodCountAnnotationKey`, `AdmissionCheck`) for
  upstream rebase.
