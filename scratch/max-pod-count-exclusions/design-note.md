# Max Pod Count Exemption — Design Note & Rollout Safeguards

Companion to `contract.md`. `contract.md` pins the in-tree behavior; this
document captures the operational contract operators must satisfy **before**
the exemption is used in production.

The in-tree code (`pkg/scheduler/framework/plugins/noderesources/fit.go`)
honours the annotation `kubelet.datadoghq.com/exclude-from-max-pods: "true"`
on any pod, in any namespace, set by any client. There is **no in-tree
policy enforcement**. Everything below — who may set the annotation, how
it's monitored, what alerts must exist — lives outside this repo, typically
in the cluster-config repo (Datadog: `k8s-resources` / equivalent).

## 1. Admission policy expectations

Cluster operators **must** install an admission policy that gates the
annotation. Two equivalent shapes are supported; pick whichever fits your
cluster's existing admission stack.

### 1a. ValidatingAdmissionPolicy (preferred)

A reference policy is shipped alongside this note as
`admission-policy.yaml`. It must:

1. Reject any CREATE/UPDATE on `pods` where the annotation is present and
   the namespace is not on an explicit allow-list.
2. Reject any UPDATE that mutates the annotation value (the annotation is
   **immutable** post-creation — flipping it on a running pod is a
   privilege escalation against the cap).
3. Reject any CREATE where the annotation is present but the request user
   / service account is not on an explicit allow-list. Service-account
   gating is the primary defence; namespace gating is the backstop.
4. Apply in `Fail` mode (not `Ignore`) so an outage in the admission stack
   does **not** silently widen the trust boundary.

The reference policy uses two `ValidatingAdmissionPolicyBinding`s — one
namespace-scoped, one service-account-scoped — so the two checks can be
tuned independently and either can be retracted in isolation during an
incident.

### 1b. Mutating webhook (alternative)

If a `ValidatingAdmissionPolicy` is not viable for the cluster (e.g. the
control plane predates CEL `params`), a mutating webhook may instead
**strip** the annotation from any pod whose namespace or service account
is not on the allow-list. This degrades to a no-op (pod is admitted, but
the cap still applies) rather than a hard reject. Trade-off: easier to
roll out, harder to audit "who tried to set this".

Either way, the property the scheduler/kubelet relies on is:

> By the time a pod reaches the noderesources plugin, the annotation is
> set **only** on pods an operator has explicitly authorised.

### 1c. What the in-tree code does NOT do

Do not expect any of the following from the kubelet/scheduler:

- No allow-list of namespaces or service accounts.
- No audit log entry on exemption use (audit must be configured in the
  apiserver, not the scheduler).
- No metric labelled by "is this pod exempt" (see §3 — operators must
  add this).
- No feature gate. The behavior is always-on; the only knob is the
  annotation. Disabling the behavior cluster-wide means denying the
  annotation in admission, not flipping a `--feature-gates` flag.

## 2. Expected impact on CNI / IPAM / node-local capacity

The kubelet pod cap (`Node.Status.Allocatable.Pods`, default 110) is one
of several independent limits on per-node pod density. Bypassing it does
**not** raise the others. Operators must verify each before enabling the
annotation on a given node pool:

| Limit                        | Where it lives                              | Symptom when saturated                       |
| ---------------------------- | ------------------------------------------- | -------------------------------------------- |
| Pod IP allocation (IPAM)     | CNI plugin (per-node CIDR slice, ENI count) | Pod stuck `ContainerCreating`, `FailedCreatePodSandBox` |
| Conntrack table              | `nf_conntrack_max` on the node              | Dropped flows, intermittent network failures |
| ARP / neighbour table        | `net.ipv4.neigh.default.gc_thresh{1,2,3}`   | Intermittent connectivity, `arp_cache: neighbor table overflow` |
| File descriptors / inotify   | `fs.file-max`, `fs.inotify.max_user_*`      | Container runtime errors, kubelet stalls     |
| PID limit                    | `kernel.pid_max`, cgroup `pids.max`         | Pod OOM-like failures, `fork: Resource temporarily unavailable` |
| cgroup hierarchy depth       | systemd / runc                              | Pod start latency, kubelet CPU spikes        |
| kube-proxy / IPVS rule count | iptables/ipvs                               | Service routing latency                      |

The CNI is by far the most common bottleneck and the one most operators
forget. Example: on AWS VPC CNI with `WARM_IP_TARGET` defaulting and ENI
limits per instance type, a `m5.large` caps at ~29 pod IPs regardless of
the kubelet `--max-pods` value. Setting the exemption on a workload
running on that node will simply move the failure from `OutOfPods` to
`FailedCreatePodSandBox: failed to assign an IP address to container`.

**Pre-flight checklist** before enabling on a node pool:

1. Confirm CNI per-node IP capacity ≥ the new effective pod count.
   For AWS VPC CNI: check ENI/IP limits for the instance type; enable
   prefix delegation if needed. For Calico/Cilium: confirm the per-node
   CIDR block size.
2. Raise `nf_conntrack_max` and `gc_thresh{1,2,3}` proportional to the
   expected new pod density.
3. Raise the kubelet `--max-pods` flag if the exemption is being used to
   pack significantly more pods than the default cap — the exemption
   does not override the kubelet's own cap (it's the scheduler-side
   admission cap that's bypassed; the kubelet still refuses to start a
   pod beyond `--max-pods` for non-exempt reasons). Make the kubelet
   cap a generous upper bound, not the operating point.
4. Confirm the container runtime (containerd / CRI-O) has cgroup, PID,
   and FD limits sized for the new density.

If any of the above is uncertain, do **not** enable the annotation on
that pool. The blast radius of CNI exhaustion is wider than the blast
radius of `OutOfPods`: `OutOfPods` is a scheduling-time signal that
fails fast and routes around the node, whereas CNI exhaustion can take
out pods that *were* scheduled successfully and are mid-restart.

## 3. Required observability & alerting

Before the annotation is used on a workload, the following must exist:

### 3a. Per-node metrics

- `kubelet_running_pods` (or equivalent kubelet metric) **as a ratio to
  `node_status_capacity{resource="pods"}`** — a per-node gauge of pod
  density. Alert when any node exceeds e.g. 90% of the kubelet cap, and
  separately when any node exceeds 90% of the **CNI** IP allocation
  (the CNI metric is provider-specific: `awscni_total_ip_addresses` for
  AWS VPC CNI, `cilium_ipam_ips_available` for Cilium, etc.).
- Conntrack utilization: `node_nf_conntrack_entries /
  node_nf_conntrack_entries_limit`. Alert at 80%.
- File descriptor utilization on the kubelet process and on the
  container runtime.

### 3b. Exemption usage visibility

The in-tree code does **not** emit a metric distinguishing exempt vs
non-exempt admissions. Operators MUST add one of:

- A scrape on the apiserver audit log filtered to
  `pods.metadata.annotations["kubelet.datadoghq.com/exclude-from-max-pods"]
  == "true"`, counted by namespace / service account / verb. This gives a
  ground-truth "who is using the exemption" view.
- A Datadog metric submitted by the admission webhook itself
  (e.g. `kubernetes.admission.pod_count_exemption.count{namespace, sa}`).
  This is cheaper but only fires on admission, not on long-lived pods.

A dashboard surfacing both is recommended for the first 90 days of any
new workload's adoption of the exemption.

### 3c. Alerts

Minimum alert set before flipping the annotation on for a workload:

1. **Node pod density** — any node above 90% of its kubelet pod cap for
   > 10 minutes. Page severity: warn.
2. **CNI IP exhaustion** — any node above 90% of its CNI per-node IP
   capacity for > 5 minutes. Page severity: high.
3. **Conntrack pressure** — any node above 80% of `nf_conntrack_max` for
   > 10 minutes. Page severity: warn.
4. **Unexpected exemption use** — non-zero count of admissions where the
   annotation was set by a namespace or service account NOT in the
   admission allow-list. Page severity: high (this means the admission
   policy is misconfigured or bypassed).
5. **Annotation churn** — non-zero count of UPDATE operations that
   mutate the annotation. The admission policy should reject these, so a
   non-zero count means the policy is in `Ignore` or `Warn` mode.
   Page severity: high.

### 3d. Datadog-specific notes

For Datadog clusters: the existing `kubernetes_state.node.allocatable`
and `kubernetes_state.node.status` integration metrics already cover the
node pod cap. The CNI per-node IP capacity is **not** covered by the
standard integration and must be wired in per-provider (DaemonSet
exporter for VPC CNI, scrape `cilium-agent` for Cilium).

## 4. Rollout plan (recommended)

1. Land the in-tree changes (steps 1-8 of `implementation-plan.md`).
2. Deploy `admission-policy.yaml` to staging clusters with an **empty**
   allow-list. Verify: any pod carrying the annotation is rejected,
   including pods authored by humans. This proves the policy is wired
   in.
3. Add a single low-stakes test namespace to the allow-list. Land a
   smoke-test workload that sets the annotation. Verify the scheduler
   admits it past the cap and verify the Datadog metric for exemption
   use fires.
4. Add the production target namespace / service account to the
   allow-list one at a time, soaking each for ≥ 24h before adding the
   next. Watch the four alerts in §3c during each soak.
5. Roll out to remaining clusters only after staging has run the
   exemption for ≥ 7 days without incident.

A rollback is a single PR removing the namespace/SA from the admission
policy allow-list — the scheduler/kubelet code does not need to be
reverted to disable the behavior cluster-wide.

## 5. Touched in-tree symbols (for cluster-config reviewers)

When the cluster-config PR references this design note, it should also
link the in-tree symbols so a reviewer chasing the contract from either
side lands in the same place:

- `pkg/scheduler/framework/plugins/noderesources/fit.go`
  - `ExcludeFromMaxPodCountAnnotationKey` (exported constant — string
    the admission policy matches on).
  - `isExcludedFromMaxPodCount` (private helper — the actual decision
    function; the godoc enumerates the same five guardrails as
    `contract.md` §4).
  - `preFilterState.ExcludeFromPodCount` (boolean computed once at
    PreFilter time).
  - `fitsRequest` (uses `+0` instead of `+1` when exempt).
- `pkg/scheduler/eventhandlers.go` — `AdmissionCheck` routes the
  kubelet through the same `Fits` path, so admission policy coverage of
  the annotation also covers kubelet admission.
- Tests in `fit_test.go`, `eventhandlers_test.go`,
  `pkg/kubelet/lifecycle/predicate_test.go` pin the
  normal-pod-blocked / exempt-pod-admitted / malformed-annotation
  fail-closed contract.

If a future change widens the exemption scope (e.g. to CPU / memory /
custom resources), this design note and the admission policy must be
updated **before** the code change lands. The contract is conservative
by design.
