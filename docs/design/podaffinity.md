<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<!-- TAG RELEASE_LINK, added by the munger automatically -->
<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.3/docs/design/podaffinity.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Inter-pod topological affinity and anti-affinity

## Introduction

NOTE: It is useful to read about [node affinity](nodeaffinity.md) first.

This document describes a proposal for specifying and implementing inter-pod
topological affinity and anti-affinity. By that we mean: rules that specify that
certain pods should be placed in the same topological domain (e.g. same node,
same rack, same zone, same power domain, etc.) as some other pods, or,
conversely, should *not* be placed in the same topological domain as some other
pods.

Here are a few example rules; we explain how to express them using the API
described in this doc later, in the section "Examples."
* Affinity
  * Co-locate the pods from a particular service or Job in the same availability
zone, without specifying which zone that should be.
  * Co-locate the pods from service S1 with pods from service S2 because S1 uses
S2 and thus it is useful to minimize the network latency between them.
Co-location might mean same nodes and/or same availability zone.
* Anti-affinity
  * Spread the pods of a service across nodes and/or availability zones, e.g. to
reduce correlated failures.
  * Give a pod "exclusive" access to a node to guarantee resource isolation --
it must never share the node with other pods.
  * Don't schedule the pods of a particular service on the same nodes as pods of
another service that are known to interfere with the performance of the pods of
the first service.

For both affinity and anti-affinity, there are three variants. Two variants have
the property of requiring the affinity/anti-affinity to be satisfied for the pod
to be allowed to schedule onto a node; the difference between them is that if
the condition ceases to be met later on at runtime, for one of them the system
will try to eventually evict the pod, while for the other the system may not try
to do so. The third variant simply provides scheduling-time *hints* that the
scheduler will try to satisfy but may not be able to. These three variants are
directly analogous to the three variants of [node affinity](nodeaffinity.md).

Note that this proposal is only about *inter-pod* topological affinity and
anti-affinity. There are other forms of topological affinity and anti-affinity.
For example, you can use [node affinity](nodeaffinity.md) to require (prefer)
that a set of pods all be scheduled in some specific zone Z. Node affinity is
not capable of expressing inter-pod dependencies, and conversely the API we
describe in this document is not capable of expressing node affinity rules. For
simplicity, we will use the terms "affinity" and "anti-affinity" to mean
"inter-pod topological affinity" and "inter-pod topological anti-affinity,"
respectively, in the remainder of this document.

## API

We will add one field to `PodSpec`

```go
Affinity *Affinity  `json:"affinity,omitempty"`
```

The `Affinity` type is defined as follows

```go
type Affinity struct {
    PodAffinity     *PodAffinity  `json:"podAffinity,omitempty"`
    PodAntiAffinity *PodAntiAffinity  `json:"podAntiAffinity,omitempty"`
}

type PodAffinity struct {
    // If the affinity requirements specified by this field are not met at
    // scheduling time, the pod will not be scheduled onto the node.
    // If the affinity requirements specified by this field cease to be met
    // at some point during pod execution (e.g. due to a pod label update), the
    // system will try to eventually evict the pod from its node.
    // When there are multiple elements, the lists of nodes corresponding to each
    // PodAffinityTerm are intersected, i.e. all terms must be satisfied.
    RequiredDuringSchedulingRequiredDuringExecution []PodAffinityTerm  `json:"requiredDuringSchedulingRequiredDuringExecution,omitempty"`
    // If the affinity requirements specified by this field are not met at
    // scheduling time, the pod will not be scheduled onto the node.
    // If the affinity requirements specified by this field cease to be met
    // at some point during pod execution (e.g. due to a pod label update), the
    // system may or may not try to eventually evict the pod from its node.
    // When there are multiple elements, the lists of nodes corresponding to each
    // PodAffinityTerm are intersected, i.e. all terms must be satisfied.
    RequiredDuringSchedulingIgnoredDuringExecution  []PodAffinityTerm  `json:"requiredDuringSchedulingIgnoredDuringExecution,omitempty"`
    // The scheduler will prefer to schedule pods to nodes that satisfy
    // the affinity expressions specified by this field, but it may choose
    // a node that violates one or more of the expressions. The node that is
    // most preferred is the one with the greatest sum of weights, i.e.
    // for each node that meets all of the scheduling requirements (resource
    // request, RequiredDuringScheduling affinity expressions, etc.),
    // compute a sum by iterating through the elements of this field and adding
    // "weight" to the sum if the node matches the corresponding MatchExpressions; the
    // node(s) with the highest sum are the most preferred.
    PreferredDuringSchedulingIgnoredDuringExecution []WeightedPodAffinityTerm  `json:"preferredDuringSchedulingIgnoredDuringExecution,omitempty"`
}

type PodAntiAffinity struct {
    // If the anti-affinity requirements specified by this field are not met at
    // scheduling time, the pod will not be scheduled onto the node.
    // If the anti-affinity requirements specified by this field cease to be met
    // at some point during pod execution (e.g. due to a pod label update), the
    // system will try to eventually evict the pod from its node.
    // When there are multiple elements, the lists of nodes corresponding to each
    // PodAffinityTerm are intersected, i.e. all terms must be satisfied.
    RequiredDuringSchedulingRequiredDuringExecution []PodAffinityTerm  `json:"requiredDuringSchedulingRequiredDuringExecution,omitempty"`
    // If the anti-affinity requirements specified by this field are not met at
    // scheduling time, the pod will not be scheduled onto the node.
    // If the anti-affinity requirements specified by this field cease to be met
    // at some point during pod execution (e.g. due to a pod label update), the
    // system may or may not try to eventually evict the pod from its node.
    // When there are multiple elements, the lists of nodes corresponding to each
    // PodAffinityTerm are intersected, i.e. all terms must be satisfied.
    RequiredDuringSchedulingIgnoredDuringExecution  []PodAffinityTerm  `json:"requiredDuringSchedulingIgnoredDuringExecution,omitempty"`
    // The scheduler will prefer to schedule pods to nodes that satisfy
    // the anti-affinity expressions specified by this field, but it may choose
    // a node that violates one or more of the expressions. The node that is
    // most preferred is the one with the greatest sum of weights, i.e.
    // for each node that meets all of the scheduling requirements (resource
    // request, RequiredDuringScheduling anti-affinity expressions, etc.),
    // compute a sum by iterating through the elements of this field and adding
    // "weight" to the sum if the node matches the corresponding MatchExpressions; the
    // node(s) with the highest sum are the most preferred.
    PreferredDuringSchedulingIgnoredDuringExecution []WeightedPodAffinityTerm  `json:"preferredDuringSchedulingIgnoredDuringExecution,omitempty"`
}

type WeightedPodAffinityTerm struct {
    // weight is in the range 1-100
    Weight int  `json:"weight"`
    PodAffinityTerm PodAffinityTerm  `json:"podAffinityTerm"`
}

type PodAffinityTerm struct {
    LabelSelector *LabelSelector `json:"labelSelector,omitempty"`
    // namespaces specifies which namespaces the LabelSelector applies to (matches against);
    // nil list means "this pod's namespace," empty list means "all namespaces"
    // The json tag here is not "omitempty" since we need to distinguish nil and empty.
    // See https://golang.org/pkg/encoding/json/#Marshal for more details.
    Namespaces []api.Namespace  `json:"namespaces,omitempty"`
    // empty topology key is interpreted by the scheduler as "all topologies"
    TopologyKey string `json:"topologyKey,omitempty"`
}
```

Note that the `Namespaces` field is necessary because normal `LabelSelector` is
scoped to the pod's namespace, but we need to be able to match against all pods
globally.

To explain how this API works, let's say that the `PodSpec` of a pod `P` has an
`Affinity` that is configured as follows (note that we've omitted and collapsed
some fields for simplicity, but this should sufficiently convey the intent of
the design):

```go
PodAffinity {
	RequiredDuringScheduling: {{LabelSelector: P1, TopologyKey: "node"}},
	PreferredDuringScheduling: {{LabelSelector: P2, TopologyKey: "zone"}},
}
PodAntiAffinity {
	RequiredDuringScheduling: {{LabelSelector: P3, TopologyKey: "rack"}},
	PreferredDuringScheduling: {{LabelSelector: P4, TopologyKey: "power"}}
}
```

Then when scheduling pod P, the scheduler:
* Can only schedule P onto nodes that are running pods that satisfy `P1`.
(Assumes all nodes have a label with key `node` and value specifying their node
name.)
* Should try to schedule P onto zones that are running pods that satisfy `P2`.
(Assumes all nodes have a label with key `zone` and value specifying their
zone.)
* Cannot schedule P onto any racks that are running pods that satisfy `P3`.
(Assumes all nodes have a label with key `rack` and value specifying their rack
name.)
* Should try not to schedule P onto any power domains that are running pods that
satisfy `P4`. (Assumes all nodes have a label with key `power` and value
specifying their power domain.)

When `RequiredDuringScheduling` has multiple elements, the requirements are
ANDed. For `PreferredDuringScheduling` the weights are added for the terms that
are satisfied for each node, and the node(s) with the highest weight(s) are the
most preferred.

In reality there are two variants of `RequiredDuringScheduling`: one suffixed
with `RequiredDuringEecution` and one suffixed with `IgnoredDuringExecution`.
For the first variant, if the affinity/anti-affinity ceases to be met at some
point during pod execution (e.g. due to a pod label update), the system will try
to eventually evict the pod from its node. In the second variant, the system may
or may not try to eventually evict the pod from its node.

## A comment on symmetry

One thing that makes affinity and anti-affinity tricky is symmetry.

Imagine a cluster that is running pods from two services, S1 and S2. Imagine
that the pods of S1 have a RequiredDuringScheduling anti-affinity rule "do not
run me on nodes that are running pods from S2." It is not sufficient just to
check that there are no S2 pods on a node when you are scheduling a S1 pod. You
also need to ensure that there are no S1 pods on a node when you are scheduling
a S2 pod, *even though the S2 pod does not have any anti-affinity rules*.
Otherwise if an S1 pod schedules before an S2 pod, the S1 pod's
RequiredDuringScheduling anti-affinity rule can be violated by a later-arriving
S2 pod. More specifically, if S1 has the aforementioned RequiredDuringScheduling
anti-affinity rule, then:
* if a node is empty, you can schedule S1 or S2 onto the node
* if a node is running S1 (S2), you cannot schedule S2 (S1) onto the node

Note that while RequiredDuringScheduling anti-affinity is symmetric,
RequiredDuringScheduling affinity is *not* symmetric. That is, if the pods of S1
have a RequiredDuringScheduling affinity rule "run me on nodes that are running
pods from S2," it is not required that there be S1 pods on a node in order to
schedule a S2 pod onto that node. More specifically, if S1 has the
aforementioned RequiredDuringScheduling affinity rule, then:
* if a node is empty, you can schedule S2 onto the node
* if a node is empty, you cannot schedule S1 onto the node
* if a node is running S2, you can schedule S1 onto the node
* if a node is running S1+S2 and S1 terminates, S2 continues running
* if a node is running S1+S2 and S2 terminates, the system terminates S1
(eventually)

However, although RequiredDuringScheduling affinity is not symmetric, there is
an implicit PreferredDuringScheduling affinity rule corresponding to every
RequiredDuringScheduling affinity rule: if the pods of S1 have a
RequiredDuringScheduling affinity rule "run me on nodes that are running pods
from S2" then it is not required that there be S1 pods on a node in order to
schedule a S2 pod onto that node, but it would be better if there are.

PreferredDuringScheduling is symmetric. If the pods of S1 had a
PreferredDuringScheduling anti-affinity rule "try not to run me on nodes that
are running pods from S2" then we would prefer to keep a S1 pod that we are
scheduling off of nodes that are running S2 pods, and also to keep a S2 pod that
we are scheduling off of nodes that are running S1 pods. Likewise if the pods of
S1 had a PreferredDuringScheduling affinity rule "try to run me on nodes that
are running pods from S2" then we would prefer to place a S1 pod that we are
scheduling onto a node that is running a S2 pod, and also to place a S2 pod that
we are scheduling onto a node that is running a S1 pod.

## Examples

Here are some examples of how you would express various affinity and
anti-affinity rules using the API we described.

### Affinity

In the examples below, the word "put" is intentionally ambiguous; the rules are
the same whether "put" means "must put" (RequiredDuringScheduling) or "try to
put" (PreferredDuringScheduling)--all that changes is which field the rule goes
into. Also, we only discuss scheduling-time, and ignore the execution-time.
Finally, some of the examples use "zone" and some use "node," just to make the
examples more interesting; any of the examples with "zone" will also work for
"node" if you change the `TopologyKey`, and vice-versa.

* **Put the pod in zone Z**:
Tricked you! It is not possible express this using the API described here. For
this you should use node affinity.

* **Put the pod in a zone that is running at least one pod from service S**:
`{LabelSelector: <selector that matches S's pods>, TopologyKey: "zone"}`

* **Put the pod on a node that is already running a pod that requires a license
for software package P**: Assuming pods that require a license for software
package P have a label `{key=license, value=P}`:
`{LabelSelector: "license" In "P", TopologyKey: "node"}`

* **Put this pod in the same zone as other pods from its same service**:
Assuming pods from this pod's service have some label `{key=service, value=S}`:
`{LabelSelector: "service" In "S", TopologyKey: "zone"}`

This last example illustrates a small issue with this API when it is used with a
scheduler that processes the pending queue one pod at a time, like the current
Kubernetes scheduler. The RequiredDuringScheduling rule
`{LabelSelector: "service" In "S", TopologyKey: "zone"}`
only "works" once one pod from service S has been scheduled. But if all pods in
service S have this RequiredDuringScheduling rule in their PodSpec, then the
RequiredDuringScheduling rule will block the first pod of the service from ever
scheduling, since it is only allowed to run in a zone with another pod from the
same service. And of course that means none of the pods of the service will be
able to schedule. This problem *only* applies to RequiredDuringScheduling
affinity, not PreferredDuringScheduling affinity or any variant of
anti-affinity. There are at least three ways to solve this problem:
* **short-term**: have the scheduler use a rule that if the
RequiredDuringScheduling affinity requirement matches a pod's own labels, and
there are no other such pods anywhere, then disregard the requirement. This
approach has a corner case when running parallel schedulers that are allowed to
schedule pods from the same replicated set (e.g. a single PodTemplate): both
schedulers may try to schedule pods from the set at the same time and think
there are no other pods from that set scheduled yet (e.g. they are trying to
schedule the first two pods from the set), but by the time the second binding is
committed, the first one has already been committed, leaving you with two pods
running that do not respect their RequiredDuringScheduling affinity. There is no
simple way to detect this "conflict" at scheduling time given the current system
implementation.
* **longer-term**: when a controller creates pods from a PodTemplate, for
exactly *one* of those pods, it should omit any RequiredDuringScheduling
affinity rules that select the pods of that PodTemplate.
* **very long-term/speculative**: controllers could present the scheduler with a
group of pods from the same PodTemplate as a single unit. This is similar to the
first approach described above but avoids the corner case. No special logic is
needed in the controllers. Moreover, this would allow the scheduler to do proper
[gang scheduling](https://github.com/kubernetes/kubernetes/issues/16845) since
it could receive an entire gang simultaneously as a single unit.

### Anti-affinity

As with the affinity examples, the examples here can be RequiredDuringScheduling
or PreferredDuringScheduling anti-affinity, i.e. "don't" can be interpreted as
"must not" or as "try not to" depending on whether the rule appears in
`RequiredDuringScheduling` or `PreferredDuringScheduling`.

* **Spread the pods of this service S across nodes and zones**:
`{{LabelSelector: <selector that matches S's pods>, TopologyKey: "node"},
{LabelSelector: <selector that matches S's pods>, TopologyKey: "zone"}}`
(note that if this is specified as a RequiredDuringScheduling anti-affinity,
then the first clause is redundant, since the second clause will force the
scheduler to not put more than one pod from S in the same zone, and thus by
definition it will not put more than one pod from S on the same node, assuming
each node is in one zone. This rule is more useful as PreferredDuringScheduling
anti-affinity, e.g. one might expect it to be common in
[Cluster Federation](../../docs/proposals/federation.md) clusters.)

* **Don't co-locate pods of this service with pods from service "evilService"**:
`{LabelSelector: selector that matches evilService's pods, TopologyKey: "node"}`

* **Don't co-locate pods of this service with any other pods including pods of this service**:
`{LabelSelector: empty, TopologyKey: "node"}`

* **Don't co-locate pods of this service with any other pods except other pods of this service**:
Assuming pods from the service have some label `{key=service, value=S}`:
`{LabelSelector: "service" NotIn "S", TopologyKey: "node"}`
Note that this works because `"service" NotIn "S"` matches pods with no key
"service" as well as pods with key "service" and a corresponding value that is
not "S."

## Algorithm

An example algorithm a scheduler might use to implement affinity and
anti-affinity rules is as follows. There are certainly more efficient ways to
do it; this is just intended to demonstrate that the API's semantics are
implementable.

Terminology definition: We say a pod P is "feasible" on a node N if P meets all
of the scheduler predicates for scheduling P onto N. Note that this algorithm is
only concerned about scheduling time, thus it makes no distinction between
RequiredDuringExecution and IgnoredDuringExecution.

To make the algorithm slightly more readable, we use the term "HardPodAffinity"
as shorthand for "RequiredDuringSchedulingScheduling pod affinity" and
"SoftPodAffinity" as shorthand for "PreferredDuringScheduling pod affinity."
Analogously for "HardPodAntiAffinity" and "SoftPodAntiAffinity."

** TODO: Update this algorithm to take weight for SoftPod{Affinity,AntiAffinity}
into account; currently it assumes all terms have weight 1. **

```
Z = the pod you are scheduling
{N} = the set of all nodes in the system  // this algorithm will reduce it to the set of all nodes feasible for Z
// Step 1a: Reduce {N} to the set of nodes satisfying Z's HardPodAffinity in the "forward" direction
X = {Z's PodSpec's HardPodAffinity}
foreach element H of {X}
	P = {all pods in the system that match H.LabelSelector}
	M map[string]int  // topology value -> number of pods running on nodes with that topology value
	foreach pod Q of {P}
		L = {labels of the node on which Q is running, represented as a map from label key to label value}
		M[L[H.TopologyKey]]++
	{N} = {N} intersect {all nodes of N with label [key=H.TopologyKey, value=any K such that M[K]>0]}
// Step 1b: Further reduce {N} to the set of nodes also satisfying Z's HardPodAntiAffinity
// This step is identical to Step 1a except the M[K] > 0 comparison becomes M[K] == 0
X = {Z's PodSpec's HardPodAntiAffinity}
foreach element H of {X}
	P = {all pods in the system that match H.LabelSelector}
	M map[string]int  // topology value -> number of pods running on nodes with that topology value
	foreach pod Q of {P}
		L = {labels of the node on which Q is running, represented as a map from label key to label value}
		M[L[H.TopologyKey]]++
	{N} = {N} intersect {all nodes of N with label [key=H.TopologyKey, value=any K such that M[K]==0]}
// Step 2: Further reduce {N} by enforcing symmetry requirement for other pods' HardPodAntiAffinity
foreach node A of {N}
	foreach pod B that is bound to A
		if any of B's HardPodAntiAffinity are currently satisfied but would be violated if Z runs on A, then remove A from {N}
// At this point, all node in {N} are feasible for Z.
// Step 3a: Soft version of Step 1a
Y map[string]int  // node -> number of Z's soft affinity/anti-affinity preferences satisfied by that node
Initialize the keys of Y to all of the nodes in {N}, and the values to 0
X = {Z's PodSpec's SoftPodAffinity}
Repeat Step 1a except replace the last line with "foreach node W of {N} having label [key=H.TopologyKey, value=any K such that M[K]>0], Y[W]++"
// Step 3b: Soft version of Step 1b
X = {Z's PodSpec's SoftPodAntiAffinity}
Repeat Step 1b except replace the last line with "foreach node W of {N} not having label [key=H.TopologyKey, value=any K such that M[K]>0], Y[W]++"
// Step 4: Symmetric soft, plus treat forward direction of hard affinity as a soft
foreach node A of {N}
	foreach pod B that is bound to A
		increment Y[A] by the number of B's SoftPodAffinity, SoftPodAntiAffinity, and HardPodAffinity that are satisfied if Z runs on A but are not satisfied if Z does not run on A
// We're done. {N} contains all of the nodes that satisfy the affinity/anti-affinity rules, and Y is
// a map whose keys are the elements of {N} and whose values are how "good" of a choice N is for Z with
// respect to the explicit and implicit affinity/anti-affinity rules (larger number is better).
```

## Special considerations for RequiredDuringScheduling anti-affinity

In this section we discuss three issues with RequiredDuringScheduling
anti-affinity: Denial of Service (DoS), co-existing with daemons, and
determining which pod(s) to kill. See issue [#18265](https://github.com/kubernetes/kubernetes/issues/18265)
for additional discussion of these topics.

### Denial of Service

Without proper safeguards, a pod using RequiredDuringScheduling anti-affinity
can intentionally or unintentionally cause various problems for other pods, due
to the symmetry property of anti-affinity.

The most notable danger is the ability for a pod that arrives first to some
topology domain, to block all other pods from scheduling there by stating a
conflict with all other pods. The standard approach to preventing resource
hogging is quota, but simple resource quota cannot prevent this scenario because
the pod may request very little resources. Addressing this using quota requires
a quota scheme that charges based on "opportunity cost" rather than based simply
on requested resources. For example, when handling a pod that expresses
RequiredDuringScheduling anti-affinity for all pods using a "node" `TopologyKey`
(i.e. exclusive access to a node), it could charge for the resources of the
average or largest node in the cluster. Likewise if a pod expresses
RequiredDuringScheduling anti-affinity for all pods using a "cluster"
`TopologyKey`, it could charge for the resources of the entire cluster. If node
affinity is used to constrain the pod to a particular topology domain, then the
admission-time quota charging should take that into account (e.g. not charge for
the average/largest machine if the PodSpec constrains the pod to a specific
machine with a known size; instead charge for the size of the actual machine
that the pod was constrained to). In all cases once the pod is scheduled, the
quota charge should be adjusted down to the actual amount of resources allocated
(e.g. the size of the actual machine that was assigned, not the
average/largest). If a cluster administrator wants to overcommit quota, for
example to allow more than N pods across all users to request exclusive node
access in a cluster with N nodes, then a priority/preemption scheme should be
added so that the most important pods run when resource demand exceeds supply.

An alternative approach, which is a bit of a blunt hammer, is to use a
capability mechanism to restrict use of RequiredDuringScheduling anti-affinity
to trusted users. A more complex capability mechanism might only restrict it
when using a non-"node" TopologyKey.

Our initial implementation will use a variant of the capability approach, which
requires no configuration: we will simply reject ALL requests, regardless of
user, that specify "all namespaces" with non-"node" TopologyKey for
RequiredDuringScheduling anti-affinity. This allows the "exclusive node" use
case while prohibiting the more dangerous ones.

A weaker variant of the problem described in the previous paragraph is a pod's
ability to use anti-affinity to degrade the scheduling quality of another pod,
but not completely block it from scheduling. For example, a set of pods S1 could
use node affinity to request to schedule onto a set of nodes that some other set
of pods S2 prefers to schedule onto. If the pods in S1 have
RequiredDuringScheduling or even PreferredDuringScheduling pod anti-affinity for
S2, then due to the symmetry property of anti-affinity, they can prevent the
pods in S2 from scheduling onto their preferred nodes if they arrive first (for
sure in the RequiredDuringScheduling case, and with some probability that
depends on the weighting scheme for the PreferredDuringScheduling case). A very
sophisticated priority and/or quota scheme could mitigate this, or alternatively
we could eliminate the symmetry property of the implementation of
PreferredDuringScheduling anti-affinity. Then only RequiredDuringScheduling
anti-affinity could affect scheduling quality of another pod, and as we
described in the previous paragraph, such pods could be charged quota for the
full topology domain, thereby reducing the potential for abuse.

We won't try to address this issue in our initial implementation; we can
consider one of the approaches mentioned above if it turns out to be a problem
in practice.

### Co-existing with daemons

A cluster administrator may wish to allow pods that express anti-affinity
against all pods, to nonetheless co-exist with system daemon pods, such as those
run by DaemonSet. In principle, we would like the specification for
RequiredDuringScheduling inter-pod anti-affinity to allow "toleration" of one or
more other pods (see [#18263](https://github.com/kubernetes/kubernetes/issues/18263)
for a more detailed explanation of the toleration concept).
There are at least two ways to accomplish this:

* Scheduler special-cases the namespace(s) where daemons live, in the
  sense that it ignores pods in those namespaces when it is
  determining feasibility for pods with anti-affinity. The name(s) of
  the special namespace(s) could be a scheduler configuration
  parameter, and default to `kube-system`. We could allow
  multiple namespaces to be specified if we want cluster admins to be
  able to give their own daemons this special power (they would add
  their namespace to the list in the scheduler configuration). And of
  course this would be symmetric, so daemons could schedule onto a node
  that is already running a pod with anti-affinity.

* We could add an explicit "toleration" concept/field to allow the
  user to specify namespaces that are excluded when they use
  RequiredDuringScheduling anti-affinity, and use an admission
  controller/defaulter to ensure these namespaces are always listed.

Our initial implementation will use the first approach.

### Determining which pod(s) to kill (for RequiredDuringSchedulingRequiredDuringExecution)

Because anti-affinity is symmetric, in the case of
RequiredDuringSchedulingRequiredDuringExecution anti-affinity, the system must
determine which pod(s) to kill when a pod's labels are updated in such as way as
to cause them to conflict with one or more other pods'
RequiredDuringSchedulingRequiredDuringExecution anti-affinity rules. In the
absence of a priority/preemption scheme, our rule will be that the pod with the
anti-affinity rule that becomes violated should be the one killed. A pod should
only specify constraints that apply to namespaces it trusts to not do malicious
things. Once we have priority/preemption, we can change the rule to say that the
lowest-priority pod(s) are killed until all
RequiredDuringSchedulingRequiredDuringExecution anti-affinity is satisfied.

## Special considerations for RequiredDuringScheduling affinity

The DoS potential of RequiredDuringScheduling *anti-affinity* stemmed from its
symmetry: if a pod P requests anti-affinity, P cannot schedule onto a node with
conflicting pods, and pods that conflict with P cannot schedule onto the node
one P has been scheduled there. The design we have described says that the
symmetry property for RequiredDuringScheduling *affinity* is weaker: if a pod P
says it can only schedule onto nodes running pod Q, this does not mean Q can
only run on a node that is running P, but the scheduler will try to schedule Q
onto a node that is running P (i.e. treats the reverse direction as preferred).
This raises the same scheduling quality concern as we mentioned at the end of
the Denial of Service section above, and can be addressed in similar ways.

The nature of affinity (as opposed to anti-affinity) means that there is no
issue of determining which pod(s) to kill when a pod's labels change: it is
obviously the pod with the affinity rule that becomes violated that must be
killed. (Killing a pod never "fixes" violation of an affinity rule; it can only
"fix" violation an anti-affinity rule.) However, affinity does have a different
question related to killing: how long should the system wait before declaring
that RequiredDuringSchedulingRequiredDuringExecution affinity is no longer met
at runtime? For example, if a pod P has such an affinity for a pod Q and pod Q
is temporarily killed so that it can be updated to a new binary version, should
that trigger killing of P? More generally, how long should the system wait
before declaring that P's affinity is violated? (Of course affinity is expressed
in terms of label selectors, not for a specific pod, but the scenario is easier
to describe using a concrete pod.) This is closely related to the concept of
forgiveness (see issue [#1574](https://github.com/kubernetes/kubernetes/issues/1574)).
In theory we could make this time duration be configurable by the user on a per-pod
basis, but for the first version of this feature we will make it a configurable
property of whichever component does the killing and that applies across all pods
using the feature. Making it configurable by the user would require a nontrivial
change to the API syntax (since the field would only apply to
RequiredDuringSchedulingRequiredDuringExecution affinity).

## Implementation plan

1. Add the `Affinity` field to PodSpec and the `PodAffinity` and
`PodAntiAffinity` types to the API along with all of their descendant types.
2. Implement a scheduler predicate that takes
`RequiredDuringSchedulingIgnoredDuringExecution` affinity and anti-affinity into
account. Include a workaround for the issue described at the end of the Affinity
section of the Examples section (can't schedule first pod).
3. Implement a scheduler priority function that takes
`PreferredDuringSchedulingIgnoredDuringExecution` affinity and anti-affinity
into account.
4. Implement admission controller that rejects requests that specify "all
namespaces" with non-"node" TopologyKey for `RequiredDuringScheduling`
anti-affinity. This admission controller should be enabled by default.
5. Implement the recommended solution to the "co-existing with daemons" issue
6. At this point, the feature can be deployed.
7. Add the `RequiredDuringSchedulingRequiredDuringExecution` field to affinity
and anti-affinity, and make sure the pieces of the system already implemented
for `RequiredDuringSchedulingIgnoredDuringExecution` also take
`RequiredDuringSchedulingRequiredDuringExecution` into account (e.g. the
scheduler predicate, the quota mechanism, the "co-existing with daemons"
solution).
8. Add `RequiredDuringSchedulingRequiredDuringExecution` for "node"
`TopologyKey` to Kubelet's admission decision.
9. Implement code in Kubelet *or* the controllers that evicts a pod that no
longer satisfies `RequiredDuringSchedulingRequiredDuringExecution`. If Kubelet
then only for "node" `TopologyKey`; if controller then potentially for all
`TopologyKeys`'s. (see [this comment](https://github.com/kubernetes/kubernetes/issues/12744#issuecomment-164372008)).
Do so in a way that addresses the "determining which pod(s) to kill" issue.

We assume Kubelet publishes labels describing the node's membership in all of
the relevant scheduling domains (e.g. node name, rack name, availability zone
name, etc.). See [#9044](https://github.com/kubernetes/kubernetes/issues/9044).

## Backward compatibility

Old versions of the scheduler will ignore `Affinity`.

Users should not start using `Affinity` until the full implementation has been
in Kubelet and the master for enough binary versions that we feel comfortable
that we will not need to roll back either Kubelet or master to a version that
does not support them. Longer-term we will use a programmatic approach to
enforcing this ([#4855](https://github.com/kubernetes/kubernetes/issues/4855)).

## Extensibility

The design described here is the result of careful analysis of use cases, a
decade of experience with Borg at Google, and a review of similar features in
other open-source container orchestration systems. We believe that it properly
balances the goal of expressiveness against the goals of simplicity and
efficiency of implementation. However, we recognize that use cases may arise in
the future that cannot be expressed using the syntax described here. Although we
are not implementing an affinity-specific extensibility mechanism for a variety
of reasons (simplicity of the codebase, simplicity of cluster deployment, desire
for Kubernetes users to get a consistent experience, etc.), the regular
Kubernetes annotation mechanism can be used to add or replace affinity rules.
The way this work would is:
1. Define one or more annotations to describe the new affinity rule(s)
1. User (or an admission controller) attaches the annotation(s) to pods to
request the desired scheduling behavior. If the new rule(s) *replace* one or
more fields of `Affinity` then the user would omit those fields from `Affinity`;
if they are *additional rules*, then the user would fill in `Affinity` as well
as the annotation(s).
1. Scheduler takes the annotation(s) into account when scheduling.

If some particular new syntax becomes popular, we would consider upstreaming it
by integrating it into the standard `Affinity`.

## Future work and non-work

One can imagine that in the anti-affinity RequiredDuringScheduling case one
might want to associate a number with the rule, for example "do not allow this
pod to share a rack with more than three other pods (in total, or from the same
service as the pod)." We could allow this to be specified by adding an integer
`Limit` to `PodAffinityTerm` just for the `RequiredDuringScheduling` case.
However, this flexibility complicates the system and we do not intend to
implement it.

It is likely that the specification and implementation of pod anti-affinity
can be unified with [taints and tolerations](taint-toleration-dedicated.md),
and likewise that the specification and implementation of pod affinity
can be unified with [node affinity](nodeaffinity.md). The basic idea is that pod
labels would be "inherited" by the node, and pods would only be able to specify
affinity and anti-affinity for a node's labels. Our main motivation for not
unifying taints and tolerations with pod anti-affinity is that we foresee taints
and tolerations as being a concept that only cluster administrators need to
understand (and indeed in some setups taints and tolerations wouldn't even be
directly manipulated by a cluster administrator, instead they would only be set
by an admission controller that is implementing the administrator's high-level
policy about different classes of special machines and the users who belong to
the groups allowed to access them). Moreover, the concept of nodes "inheriting"
labels from pods seems complicated; it seems conceptually simpler to separate
rules involving relatively static properties of nodes from rules involving which
other pods are running on the same node or larger topology domain.

Data/storage affinity is related to pod affinity, and is likely to draw on some
of the ideas we have used for pod affinity. Today, data/storage affinity is
expressed using node affinity, on the assumption that the pod knows which
node(s) store(s) the data it wants. But a more flexible approach would allow the
pod to name the data rather than the node.

## Related issues

The review for this proposal is in [#18265](https://github.com/kubernetes/kubernetes/issues/18265).

The topic of affinity/anti-affinity has generated a lot of discussion. The main
issue is [#367](https://github.com/kubernetes/kubernetes/issues/367)
but [#14484](https://github.com/kubernetes/kubernetes/issues/14484)/[#14485](https://github.com/kubernetes/kubernetes/issues/14485),
[#9560](https://github.com/kubernetes/kubernetes/issues/9560), [#11369](https://github.com/kubernetes/kubernetes/issues/11369),
[#14543](https://github.com/kubernetes/kubernetes/issues/14543), [#11707](https://github.com/kubernetes/kubernetes/issues/11707),
[#3945](https://github.com/kubernetes/kubernetes/issues/3945), [#341](https://github.com/kubernetes/kubernetes/issues/341),
[#1965](https://github.com/kubernetes/kubernetes/issues/1965), and [#2906](https://github.com/kubernetes/kubernetes/issues/2906)
all have additional discussion and use cases.

As the examples in this document have demonstrated, topological affinity is very
useful in clusters that are spread across availability zones, e.g. to co-locate
pods of a service in the same zone to avoid a wide-area network hop, or to
spread pods across zones for failure tolerance. [#17059](https://github.com/kubernetes/kubernetes/issues/17059),
[#13056](https://github.com/kubernetes/kubernetes/issues/13056), [#13063](https://github.com/kubernetes/kubernetes/issues/13063),
and [#4235](https://github.com/kubernetes/kubernetes/issues/4235) are relevant.

Issue [#15675](https://github.com/kubernetes/kubernetes/issues/15675) describes connection affinity, which is vaguely related.

This proposal is to satisfy [#14816](https://github.com/kubernetes/kubernetes/issues/14816).

## Related work

** TODO: cite references **



<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/podaffinity.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
