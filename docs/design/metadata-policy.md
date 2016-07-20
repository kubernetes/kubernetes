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
[here](http://releases.k8s.io/release-1.3/docs/design/metadata-policy.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# MetadataPolicy and its use in choosing the scheduler in a multi-scheduler system

## Introduction

This document describes a new API resource, `MetadataPolicy`, that configures an
admission controller to take one or more actions based on an object's metadata.
Initially the metadata fields that the predicates can examine are labels and
annotations, and the actions are to add one or more labels and/or annotations,
or to reject creation/update of the object. In the future other actions might be
supported, such as applying an initializer.

The first use of `MetadataPolicy` will be to decide which scheduler should
schedule a pod in a [multi-scheduler](../proposals/multiple-schedulers.md)
Kubernetes system. In particular, the policy will add the scheduler name
annotation to a pod based on an annotation that is already on the pod that
indicates the QoS of the pod. (That annotation was presumably set by a simpler
admission controller that uses code, rather than configuration, to map the
resource requests and limits of a pod to QoS, and attaches the corresponding
annotation.)

We anticipate a  number of other uses for `MetadataPolicy`, such as defaulting
for labels and annotations, prohibiting/requiring particular labels or
annotations, or choosing a scheduling policy within a scheduler. We do not
discuss them in this doc.


## API

```go
// MetadataPolicySpec defines the configuration of the MetadataPolicy API resource.
// Every rule is applied, in an unspecified order, but if the action for any rule
// that matches is to reject the object, then the object is rejected without being mutated.
type MetadataPolicySpec struct {
	Rules []MetadataPolicyRule `json:"rules,omitempty"`
}

// If the PolicyPredicate is met, then the PolicyAction is applied.
// Example rules:
//    reject object if label with key X is present (i.e. require X)
//    reject object if label with key X is not present (i.e. forbid X)
//    add label X=Y if label with key X is not present (i.e. default X)
//    add annotation A=B if object has annotation C=D or E=F
type MetadataPolicyRule struct {
	PolicyPredicate PolicyPredicate `json:"policyPredicate"`
	PolicyAction PolicyAction `json:policyAction"`
}

// All criteria must be met for the PolicyPredicate to be considered met.
type PolicyPredicate struct {
	// Note that Namespace is not listed here because MetadataPolicy is per-Namespace.	
	LabelSelector *LabelSelector       `json:"labelSelector,omitempty"`
	AnnotationSelector *LabelSelector  `json:"annotationSelector,omitempty"`
}

// Apply the indicated Labels and/or Annotations (if present), unless Reject is set
// to true, in which case reject the object without mutating it.
type PolicyAction struct {
	// If true, the object will be rejected and not mutated.
	Reject bool `json:"reject"`
	// The labels to add or update, if any.
	UpdatedLabels *map[string]string `json:"updatedLabels,omitempty"`
	// The annotations to add or update, if any.
	UpdatedAnnotations *map[string]string `json:"updatedAnnotations,omitempty"`
}

// MetadataPolicy describes the MetadataPolicy API resource, which is used for specifying
// policies that should be applied to objects based on the objects' metadata. All MetadataPolicy's
// are applied to all objects in the namespace; the order of evaluation is not guaranteed,
// but if any of the matching policies have an action of rejecting the object, then the object
// will be rejected without being mutated.
type MetadataPolicy struct {
	unversioned.TypeMeta `json:",inline"`
	// Standard object's metadata.
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#metadata
	ObjectMeta `json:"metadata,omitempty"`

	// Spec defines the metadata policy that should be enforced.
	// http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#spec-and-status
	Spec MetadataPolicySpec `json:"spec,omitempty"`
}

// MetadataPolicyList is a list of MetadataPolicy items.
type MetadataPolicyList struct {
	unversioned.TypeMeta `json:",inline"`
	// Standard list metadata.
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#types-kinds
	unversioned.ListMeta `json:"metadata,omitempty"`

	// Items is a list of MetadataPolicy objects.
	// More info: http://releases.k8s.io/HEAD/docs/design/admission_control_resource_quota.md#admissioncontrol-plugin-resourcequota
	Items []MetadataPolicy `json:"items"`
}
```

## Implementation plan

1. Create `MetadataPolicy` API resource
1. Create admission controller that implements policies defined in
`MetadataPolicy`
1. Create admission controller that sets annotation
`scheduler.alpha.kubernetes.io/qos: <QoS>`
(where `QOS` is one of `Guaranteed, Burstable, BestEffort`)
based on pod's resource request and limit.

## Future work

Longer-term we will have QoS be set on create and update by the registry,
similar to `Pending` phase today, instead of having an admission controller
(that runs before the one that takes `MetadataPolicy` as input) do it.

We plan to eventually move from having an admission controller set the scheduler
name as a pod annotation, to using the initializer concept. In particular, the
scheduler will be an initializer, and the admission controller that decides
which scheduler to use will add the scheduler's name to the list of initializers
for the pod (presumably the scheduler will be the last initializer to run on
each pod). The admission controller would still be configured using the
`MetadataPolicy` described here, only the mechanism the admission controller
uses to record its decision of which scheduler to use would change.

## Related issues

The main issue for multiple schedulers is #11793. There was also a lot of
discussion in PRs #17197 and #17865.

We could use the approach described here to choose a scheduling policy within a
single scheduler, as opposed to choosing a scheduler, a desire mentioned in

# 9920. Issue #17097 describes a scenario unrelated to scheduler-choosing where

`MetadataPolicy` could be used. Issue #17324 proposes to create a generalized
API for matching "claims" to "service classes"; matching a pod to a scheduler
would be one use for such an API.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/metadata-policy.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
