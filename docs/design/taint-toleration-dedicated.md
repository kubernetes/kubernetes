# Taints, Tolerations, and Dedicated Nodes

## Introduction

This document describes *taints* and *tolerations*, which constitute a generic
mechanism for restricting the set of pods that can use a node. We also describe
one concrete use case for the mechanism, namely to limit the set of users (or
more generally, authorization domains) who can access a set of nodes (a feature
we call *dedicated nodes*). There are many other uses--for example, a set of
nodes with a particular piece of hardware could be reserved for pods that
require that hardware, or a node could be marked as unschedulable when it is
being drained before shutdown, or a node could trigger evictions when it
experiences hardware or software problems or abnormal node configurations; see
issues [#17190](https://github.com/kubernetes/kubernetes/issues/17190) and
[#3885](https://github.com/kubernetes/kubernetes/issues/3885) for more discussion.

## Taints, tolerations, and dedicated nodes

A *taint* is a new type that is part of the `NodeSpec`; when present, it
prevents pods from scheduling onto the node unless the pod *tolerates* the taint
(tolerations are listed in the `PodSpec`). Note that there are actually multiple
flavors of taints: taints that prevent scheduling on a node, taints that cause
the scheduler to try to avoid scheduling on a node but do not prevent it, taints
that prevent a pod from starting on Kubelet even if the pod's `NodeName` was
written directly (i.e. pod did not go through the scheduler), and taints that
evict already-running pods.
[This comment](https://github.com/kubernetes/kubernetes/issues/3885#issuecomment-146002375)
has more background on these different scenarios. We will focus on the first
kind of taint in this doc, since it is the kind required for the "dedicated
nodes" use case.

Implementing dedicated nodes using taints and tolerations is straightforward: in
essence, a node that is dedicated to group A gets taint `dedicated=A` and the
pods belonging to group A get toleration `dedicated=A`. (The exact syntax and
semantics of taints and tolerations are described later in this doc.) This keeps
all pods except those belonging to group A off of the nodes. This approach
easily generalizes to pods that are allowed to schedule into multiple dedicated
node groups, and nodes that are a member of multiple dedicated node groups.

Note that because tolerations are at the granularity of pods, the mechanism is
very flexible -- any policy can be used to determine which tolerations should be
placed on a pod. So the "group A" mentioned above could be all pods from a
particular namespace or set of namespaces, or all pods with some other arbitrary
characteristic in common. We expect that any real-world usage of taints and
tolerations will employ an admission controller to apply the tolerations. For
example, to give all pods from namespace A access to dedicated node group A, an
admission controller would add the corresponding toleration to all pods from
namespace A. Or to give all pods that require GPUs access to GPU nodes, an
admission controller would add the toleration for GPU taints to pods that
request the GPU resource.

Everything that can be expressed using taints and tolerations can be expressed
using [node affinity](https://github.com/kubernetes/kubernetes/pull/18261), e.g.
in the example in the previous paragraph, you could put a label `dedicated=A` on
the set of dedicated nodes and a node affinity `dedicated NotIn A` on all pods *not*
belonging to group A. But it is cumbersome to express exclusion policies using
node affinity because every time you add a new type of restricted node, all pods
that aren't allowed to use those nodes need to start avoiding those nodes using
node affinity. This means the node affinity list can get quite long in clusters
with lots of different groups of special nodes (lots of dedicated node groups,
lots of different kinds of special hardware, etc.). Moreover, you need to also
update any Pending pods when you add new types of special nodes. In contrast,
with taints and tolerations, when you add a new type of special node, "regular"
pods are unaffected, and you just need to add the necessary toleration to the
pods you subsequent create that need to use the new type of special nodes. To
put it another way, with taints and tolerations, only pods that use a set of
special nodes need to know about those special nodes; with the node affinity
approach, pods that have no interest in those special nodes need to know about
all of the groups of special nodes.

One final comment: in practice, it is often desirable to not only keep "regular"
pods off of special nodes, but also to keep "special" pods off of regular nodes.
An example in the dedicated nodes case is to not only keep regular users off of
dedicated nodes, but also to keep dedicated users off of non-dedicated (shared)
nodes. In this case, the "non-dedicated" nodes can be modeled as their own
dedicated node group (for example, tainted as `dedicated=shared`), and pods that
are not given access to any dedicated nodes ("regular" pods) would be given a
toleration for `dedicated=shared`. (As mentioned earlier, we expect tolerations
will be added by an admission controller.) In this case taints/tolerations are
still better than node affinity because with taints/tolerations each pod only
needs one special "marking", versus in the node affinity case where every time
you add a dedicated node group (i.e. a new `dedicated=` value), you need to add
a new node affinity rule to all pods (including pending pods) except the ones
allowed to use that new dedicated node group.

## API

```go
// The node this Taint is attached to has the effect "effect" on
// any pod that that does not tolerate the Taint.
type Taint struct {
  Key string  `json:"key" patchStrategy:"merge" patchMergeKey:"key"`
  Value string  `json:"value,omitempty"`
  Effect TaintEffect  `json:"effect"`
}

type TaintEffect string

const (
  // Do not allow new pods to schedule unless they tolerate the taint,
  // but allow all pods submitted to Kubelet without going through the scheduler
  // to start, and allow all already-running pods to continue running. 
  // Enforced by the scheduler.
  TaintEffectNoSchedule TaintEffect = "NoSchedule"
  // Like TaintEffectNoSchedule, but the scheduler tries not to schedule
  // new pods onto the node, rather than prohibiting new pods from scheduling
  // onto the node. Enforced by the scheduler.
  TaintEffectPreferNoSchedule TaintEffect = "PreferNoSchedule"
  // Do not allow new pods to schedule unless they tolerate the taint,
  // do not allow pods to start on Kubelet unless they tolerate the taint,
  // but allow all already-running pods to continue running.
  // Enforced by the scheduler and Kubelet.
  TaintEffectNoScheduleNoAdmit TaintEffect = "NoScheduleNoAdmit"
  // Do not allow new pods to schedule unless they tolerate the taint,
  // do not allow pods to start on Kubelet unless they tolerate the taint,
  // and try to eventually evict any already-running pods that do not tolerate the taint.
  // Enforced by the scheduler and Kubelet.
  TaintEffectNoScheduleNoAdmitNoExecute = "NoScheduleNoAdmitNoExecute"
)

// The pod this Toleration is attached to tolerates any taint that matches
// the triple <key,value,effect> using the matching operator <operator>.
type Toleration struct {
  Key string  `json:"key" patchStrategy:"merge" patchMergeKey:"key"`
  // operator represents a key's relationship to the value.
  // Valid operators are Exists and Equal. Defaults to Equal.
  // Exists is equivalent to wildcard for value, so that a pod can
  // tolerate all taints of a particular category.
  Operator TolerationOperator `json:"operator"`
  Value string                `json:"value,omitempty"`
  Effect TaintEffect          `json:"effect"`
  // TODO: For forgiveness (#1574), we'd eventually add at least a grace period
  // here, and possibly an occurrence threshold and period.
}

// A toleration operator is the set of operators that can be used in a toleration.
type TolerationOperator string

const (
  TolerationOpExists  TolerationOperator = "Exists"
  TolerationOpEqual   TolerationOperator = "Equal"
)

```

(See [this comment](https://github.com/kubernetes/kubernetes/issues/3885#issuecomment-146002375)
to understand the motivation for the various taint effects.)

We will add:

```go
	// Multiple tolerations with the same key are allowed.
	Tolerations []Toleration  `json:"tolerations,omitempty"`
```

to `PodSpec`. A pod must tolerate all of a node's taints (except taints of type
TaintEffectPreferNoSchedule) in order to be able to schedule onto that node.

We will add:

```go
	// Multiple taints with the same key are not allowed.
	Taints []Taint  `json:"taints,omitempty"`
```

to both `NodeSpec` and `NodeStatus`. The value in `NodeStatus` is the union
of the taints specified by various sources. For now, the only source is
the `NodeSpec` itself, but in the future one could imagine a node inheriting
taints from pods (if we were to allow taints to be attached to pods), from
the node's startup configuration, etc. The scheduler should look at the `Taints`
in `NodeStatus`, not in `NodeSpec`.

Taints and tolerations are not scoped to namespace.

## Implementation plan: taints, tolerations, and dedicated nodes

Using taints and tolerations to implement dedicated nodes requires these steps:

1. Add the API described above
1. Add a scheduler predicate function that respects taints and tolerations (for
TaintEffectNoSchedule) and a scheduler priority function that respects taints
and tolerations (for TaintEffectPreferNoSchedule).
1. Add to the Kubelet code to implement the "no admit" behavior of
TaintEffectNoScheduleNoAdmit and TaintEffectNoScheduleNoAdmitNoExecute
1. Implement code in Kubelet that evicts a pod that no longer satisfies
TaintEffectNoScheduleNoAdmitNoExecute. In theory we could do this in the
controllers instead, but since taints might be used to enforce security
policies, it is better to do in kubelet because kubelet can respond quickly and
can guarantee the rules will be applied to all pods. Eviction may need to happen
under a variety of circumstances: when a taint is added, when an existing taint
is updated, when a toleration is removed from a pod, or when a toleration is
modified on a pod.
1. Add a new `kubectl` command that adds/removes taints to/from nodes,
1. (This is the one step is that is specific to dedicated nodes) Implement an
admission controller that adds tolerations to pods that are supposed to be
allowed to use dedicated nodes (for example, based on pod's namespace).

In the future one can imagine a generic policy configuration that configures an
admission controller to apply the appropriate tolerations to the desired class
of pods and taints to Nodes upon node creation. It could be used not just for
policies about dedicated nodes, but also other uses of taints and tolerations,
e.g. nodes that are restricted due to their hardware configuration.

The `kubectl` command to add and remove taints on nodes will be modeled after
`kubectl label`. Examples usages:

```sh
# Update node 'foo' with a taint with key 'dedicated' and value 'special-user' and effect 'NoScheduleNoAdmitNoExecute'.
# If a taint with that key already exists, its value and effect are replaced as specified.
$ kubectl taint nodes foo dedicated=special-user:NoScheduleNoAdmitNoExecute

# Remove from node 'foo' the taint with key 'dedicated' if one exists.
$ kubectl taint nodes foo dedicated-
```

## Example: implementing a dedicated nodes policy

Let's say that the cluster administrator wants to make nodes `foo`, `bar`, and `baz` available
only to pods in a particular namespace `banana`. First the administrator does

```sh
$ kubectl taint nodes foo dedicated=banana:NoScheduleNoAdmitNoExecute
$ kubectl taint nodes bar dedicated=banana:NoScheduleNoAdmitNoExecute
$ kubectl taint nodes baz dedicated=banana:NoScheduleNoAdmitNoExecute

```

(assuming they want to evict pods that are already running on those nodes if those
pods don't already tolerate the new taint)

Then they ensure that the `PodSpec` for all pods created in namespace `banana` specify
a toleration with `key=dedicated`, `value=banana`, and `policy=NoScheduleNoAdmitNoExecute`.

In the future, it would be nice to be able to specify the nodes via a `NodeSelector` rather than having
to enumerate them by name.

## Future work

At present, the Kubernetes security model allows any user to add and remove any
taints and tolerations. Obviously this makes it impossible to securely enforce
rules like dedicated nodes. We need some mechanism that prevents regular users
from mutating the `Taints` field of `NodeSpec` (probably we want to prevent them
from mutating any fields of `NodeSpec`) and from mutating the `Tolerations`
field of their pods. [#17549](https://github.com/kubernetes/kubernetes/issues/17549)
is relevant.

Another security vulnerability arises if nodes are added to the cluster before
receiving their taint. Thus we need to ensure that a new node does not become
"Ready" until it has been configured with its taints. One way to do this is to
have an admission controller that adds the taint whenever a Node object is
created.

A quota policy may want to treat nodes differently based on what taints, if any,
they have. For example, if a particular namespace is only allowed to access
dedicated nodes, then it may be convenient to give the namespace unlimited
quota. (To use finite quota, you'd have to size the namespace's quota to the sum
of the sizes of the machines in the dedicated node group, and update it when
nodes are added/removed to/from the group.)

It's conceivable that taints and tolerations could be unified with
[pod anti-affinity](https://github.com/kubernetes/kubernetes/pull/18265).
We have chosen not to do this for the reasons described in the "Future work"
section of that doc.

## Backward compatibility

Old scheduler versions will ignore taints and tolerations. New scheduler
versions will respect them.

Users should not start using taints and tolerations until the full
implementation has been in Kubelet and the master for enough binary versions
that we feel comfortable that we will not need to roll back either Kubelet or
master to a version that does not support them. Longer-term we will use a
programatic approach to enforcing this ([#4855](https://github.com/kubernetes/kubernetes/issues/4855)).

## Related issues

This proposal is based on the discussion in [#17190](https://github.com/kubernetes/kubernetes/issues/17190).
There are a number of other related issues, all of which are linked to from
[#17190](https://github.com/kubernetes/kubernetes/issues/17190).

The relationship between taints and node drains is discussed in [#1574](https://github.com/kubernetes/kubernetes/issues/1574).

The concepts of taints and tolerations were originally developed as part of the
Omega project at Google.



<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/taint-toleration-dedicated.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
