# Node affinity and NodeSelector

## Introduction

This document proposes a new label selector representation, called
`NodeSelector`, that is similar in many ways to `LabelSelector`, but is a bit
more flexible and is intended to be used only for selecting nodes.

In addition, we propose to replace the `map[string]string` in `PodSpec` that the
scheduler currently uses as part of restricting the set of nodes onto which a
pod is eligible to schedule, with a field of type `Affinity` that contains one
or more affinity specifications. In this document we discuss `NodeAffinity`,
which contains one or more of the following:
* a field called `RequiredDuringSchedulingRequiredDuringExecution` that will be
represented by a `NodeSelector`, and thus generalizes the scheduling behavior of
the current `map[string]string` but still serves the purpose of restricting
the set of nodes onto which the pod can schedule. In addition, unlike the
behavior of the current `map[string]string`, when it becomes violated the system
will try to eventually evict the pod from its node.
* a field called `RequiredDuringSchedulingIgnoredDuringExecution` which is
identical to `RequiredDuringSchedulingRequiredDuringExecution` except that the
system may or may not try to eventually evict the pod from its node.
* a field called `PreferredDuringSchedulingIgnoredDuringExecution` that
specifies which nodes are preferred for scheduling among those that meet all
scheduling requirements.

(In practice, as discussed later, we will actually *add* the `Affinity` field
rather than replacing `map[string]string`, due to backward compatibility
requirements.)

The affinity specifications described above allow a pod to request various
properties that are inherent to nodes, for example "run this pod on a node with
an Intel CPU" or, in a multi-zone cluster, "run this pod on a node in zone Z."
([This issue](https://github.com/kubernetes/kubernetes/issues/9044) describes
some of the properties that a node might publish as labels, which affinity
expressions can match against.) They do *not* allow a pod to request to schedule
(or not schedule) on a node based on what other pods are running on the node.
That feature is called "inter-pod topological affinity/anti-affinity" and is
described [here](https://github.com/kubernetes/kubernetes/pull/18265).

## API

### NodeSelector

```go
// A node selector represents the union of the results of one or more label queries
// over a set of nodes; that is, it represents the OR of the selectors represented
// by the nodeSelectorTerms.
type NodeSelector struct {
	// nodeSelectorTerms is a list of node selector terms. The terms are ORed.
	NodeSelectorTerms []NodeSelectorTerm `json:"nodeSelectorTerms,omitempty"`
}

// An empty node selector term matches all objects. A null node selector term
// matches no objects.
type NodeSelectorTerm struct {
	// matchExpressions is a list of node selector requirements. The requirements are ANDed.
	MatchExpressions []NodeSelectorRequirement `json:"matchExpressions,omitempty"`
}

// A node selector requirement is a selector that contains values, a key, and an operator
// that relates the key and values.
type NodeSelectorRequirement struct {
	// key is the label key that the selector applies to.
	Key string `json:"key" patchStrategy:"merge" patchMergeKey:"key"`
	// operator represents a key's relationship to a set of values.
	// Valid operators are In, NotIn, Exists, DoesNotExist. Gt, and Lt.
	Operator NodeSelectorOperator `json:"operator"`
	// values is an array of string values. If the operator is In or NotIn,
	// the values array must be non-empty. If the operator is Exists or DoesNotExist,
	// the values array must be empty. If the operator is Gt or Lt, the values
	// array must have a single element, which will be interpreted as an integer.
    // This array is replaced during a strategic merge patch.
	Values []string `json:"values,omitempty"`
}

// A node selector operator is the set of operators that can be used in
// a node selector requirement.
type NodeSelectorOperator string

const (
	NodeSelectorOpIn           NodeSelectorOperator = "In"
	NodeSelectorOpNotIn        NodeSelectorOperator = "NotIn"
	NodeSelectorOpExists       NodeSelectorOperator = "Exists"
	NodeSelectorOpDoesNotExist NodeSelectorOperator = "DoesNotExist"
	NodeSelectorOpGt           NodeSelectorOperator = "Gt"
	NodeSelectorOpLt           NodeSelectorOperator = "Lt"
)
```

### NodeAffinity

We will add one field to `PodSpec`

```go
Affinity *Affinity  `json:"affinity,omitempty"`
```

The `Affinity` type is defined as follows

```go
type Affinity struct {
	NodeAffinity *NodeAffinity `json:"nodeAffinity,omitempty"`
}

type NodeAffinity struct {
	// If the affinity requirements specified by this field are not met at
	// scheduling time, the pod will not be scheduled onto the node.
	// If the affinity requirements specified by this field cease to be met
	// at some point during pod execution (e.g. due to a node label update),
	// the system will try to eventually evict the pod from its node.
	RequiredDuringSchedulingRequiredDuringExecution *NodeSelector  `json:"requiredDuringSchedulingRequiredDuringExecution,omitempty"`
	// If the affinity requirements specified by this field are not met at
	// scheduling time, the pod will not be scheduled onto the node.
	// If the affinity requirements specified by this field cease to be met
	// at some point during pod execution (e.g. due to a node label update),
	// the system may or may not try to eventually evict the pod from its node.
	RequiredDuringSchedulingIgnoredDuringExecution  *NodeSelector  `json:"requiredDuringSchedulingIgnoredDuringExecution,omitempty"`
	// The scheduler will prefer to schedule pods to nodes that satisfy
	// the affinity expressions specified by this field, but it may choose
	// a node that violates one or more of the expressions. The node that is
	// most preferred is the one with the greatest sum of weights, i.e.
	// for each node that meets all of the scheduling requirements (resource
	// request, RequiredDuringScheduling affinity expressions, etc.),
	// compute a sum by iterating through the elements of this field and adding
	// "weight" to the sum if the node matches the corresponding MatchExpressions; the
	// node(s) with the highest sum are the most preferred.
	PreferredDuringSchedulingIgnoredDuringExecution []PreferredSchedulingTerm  `json:"preferredDuringSchedulingIgnoredDuringExecution,omitempty"`
}

// An empty preferred scheduling term matches all objects with implicit weight 0
// (i.e. it's a no-op). A null preferred scheduling term matches no objects.
type PreferredSchedulingTerm struct {
    // weight is in the range 1-100
	Weight int  `json:"weight"`
	// matchExpressions is a list of node selector requirements. The requirements are ANDed.
	MatchExpressions []NodeSelectorRequirement  `json:"matchExpressions,omitempty"`
}
```

Unfortunately, the name of the existing `map[string]string` field in PodSpec is
`NodeSelector` and we can't change it since this name is part of the API.
Hopefully this won't cause too much confusion.

## Examples

** TODO: fill in this section **

* Run this pod on a node with an Intel or AMD CPU

* Run this pod on a node in availability zone Z


## Backward compatibility

When we add `Affinity` to PodSpec, we will deprecate, but not remove, the
current field in PodSpec

```go
NodeSelector map[string]string `json:"nodeSelector,omitempty"`
```

Old version of the scheduler will ignore the `Affinity` field. New versions of
the scheduler will apply their scheduling predicates to both `Affinity` and
`nodeSelector`, i.e. the pod can only schedule onto nodes that satisfy both sets
of requirements. We will not attempt to convert between `Affinity` and
`nodeSelector`.

Old versions of non-scheduling clients will not know how to do anything
semantically meaningful with `Affinity`, but we don't expect that this will
cause a problem.

See [this comment](https://github.com/kubernetes/kubernetes/issues/341#issuecomment-140809259)
for more discussion.

Users should not start using `NodeAffinity` until the full implementation has
been in Kubelet and the master for enough binary versions that we feel
comfortable that we will not need to roll back either Kubelet or master to a
version that does not support them. Longer-term we will use a programatic
approach to enforcing this ([#4855](https://github.com/kubernetes/kubernetes/issues/4855)).

## Implementation plan

1. Add the `Affinity` field to PodSpec and the `NodeAffinity`,
`PreferredDuringSchedulingIgnoredDuringExecution`, and
`RequiredDuringSchedulingIgnoredDuringExecution` types to the API.
2. Implement a scheduler predicate that takes
`RequiredDuringSchedulingIgnoredDuringExecution` into account.
3. Implement a scheduler priority function that takes
`PreferredDuringSchedulingIgnoredDuringExecution` into account.
4. At this point, the feature can be deployed and `PodSpec.NodeSelector` can be
marked as deprecated.
5. Add the `RequiredDuringSchedulingRequiredDuringExecution` field to the API.
6. Modify the scheduler predicate from step 2 to also take
`RequiredDuringSchedulingRequiredDuringExecution` into account.
7. Add `RequiredDuringSchedulingRequiredDuringExecution` to Kubelet's admission
decision.
8. Implement code in Kubelet *or* the controllers that evicts a pod that no
longer satisfies `RequiredDuringSchedulingRequiredDuringExecution` (see [this comment](https://github.com/kubernetes/kubernetes/issues/12744#issuecomment-164372008)).

We assume Kubelet publishes labels describing the node's membership in all of
the relevant scheduling domains (e.g. node name, rack name, availability zone
name, etc.). See [#9044](https://github.com/kubernetes/kubernetes/issues/9044).

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

## Future work

Are there any other fields we should convert from `map[string]string` to
`NodeSelector`?

## Related issues

The review for this proposal is in [#18261](https://github.com/kubernetes/kubernetes/issues/18261).

The main related issue is [#341](https://github.com/kubernetes/kubernetes/issues/341).
Issue [#367](https://github.com/kubernetes/kubernetes/issues/367) is also related.
Those issues reference other related issues.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/nodeaffinity.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
