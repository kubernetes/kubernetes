/*
Copyright 2019 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package v1

import (
	apiv1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.14

// PriorityClass defines mapping from a priority class name to the priority
// integer value. The value can be any valid integer.
type PriorityClass struct {
	metav1.TypeMeta `json:""`
	// metadata is the standard object metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// value represents the integer value of this priority class. This is the actual priority that pods
	// receive when they have the name of this class in their pod spec.
	// +optional
	Value int32 `json:"value" protobuf:"bytes,2,opt,name=value"`

	// globalDefault specifies whether this PriorityClass should be considered as
	// the default priority for pods that do not have any priority class.
	// Only one PriorityClass can be marked as `globalDefault`. However, if more than
	// one PriorityClasses exists with their `globalDefault` field set to true,
	// the smallest value of such global default PriorityClasses will be used as the default priority.
	// +optional
	GlobalDefault bool `json:"globalDefault,omitempty" protobuf:"bytes,3,opt,name=globalDefault"`

	// description is an arbitrary string that usually provides guidelines on
	// when this priority class should be used.
	// +optional
	Description string `json:"description,omitempty" protobuf:"bytes,4,opt,name=description"`

	// preemptionPolicy is the Policy for preempting pods with lower priority.
	// One of Never, PreemptLowerPriority.
	// Defaults to PreemptLowerPriority if unset.
	// +optional
	PreemptionPolicy *apiv1.PreemptionPolicy `json:"preemptionPolicy,omitempty" protobuf:"bytes,5,opt,name=preemptionPolicy"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.14

// PriorityClassList is a collection of priority classes.
type PriorityClassList struct {
	metav1.TypeMeta `json:""`
	// Standard list metadata
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// items is the list of PriorityClasses
	Items []PriorityClass `json:"items" protobuf:"bytes,2,rep,name=items"`
}

// WorkloadTopologyConstraint defines a topology constraint for a group of
// pods managed by a workload controller.
type WorkloadTopologyConstraint struct {
	// key specifies the key of the node label representing the topology domain.
	// All pods within the PodGroup must be colocated within the same domain instance.
	// Different PodGroups can land on different domain instances even if they derive from the same PodGroupTemplate.
	// Nodes where this label is missing or has an empty value do not belong to any domain instance,
	// so pods within the PodGroup cannot be scheduled onto them.
	// Examples: "topology.kubernetes.io/rack"
	//
	// +required
	// +k8s:required
	// +k8s:format=k8s-label-key
	Key string `json:"key" protobuf:"bytes,1,opt,name=key"`
}

// WorkloadPodGroupSchedulingPolicy defines the scheduling policy for a
// group of pods managed by a workload controller.
// Exactly one policy must be set.
//
// ---
//
// This is a reusable building block meant to be embedded in a controller's own
// API, next to its other scheduling fields (for example, in a Job's
// spec.scheduling). It's recommended to block changing the policy
// after creation, while still allowing gang.minCount to change. DV cannot
// express that only gang.minCount is mutable while the basic/gang variant is
// frozen, so the embedder must enforce variant immutability with hand-written
// validation. For example,
//
//	type JobSchedulingConfiguration struct {
//		// SchedulingPolicy defines the scheduling policy for this Job.
//		// Exactly one of Basic or Gang must be set.
//		// This field is immutable after creation: the policy may not be added or
//		// removed. The policy variant (basic/gang) is frozen by the controller's
//		// hand-written validation; only schedulingPolicy.gang.minCount may be changed.
//		// +optional
//		// +k8s:optional
//		// +k8s:update=NoSet
//		// +k8s:update=NoUnset
//		SchedulingPolicy *schedulingv1.WorkloadPodGroupSchedulingPolicy `json:"schedulingPolicy,omitempty" protobuf:"bytes,N,opt,name=schedulingPolicy"`
//
//		// other scheduling fields
//	}
//
// +union
type WorkloadPodGroupSchedulingPolicy struct {
	// basic specifies that standard, pod-by-pod Kubernetes scheduling
	// behavior should be used.
	//
	// +optional
	// +k8s:optional
	// +k8s:unionMember
	Basic *WorkloadPodGroupBasicSchedulingPolicy `json:"basic,omitempty" protobuf:"bytes,1,opt,name=basic"`

	// gang specifies all-or-nothing scheduling semantics.
	//
	// +optional
	// +k8s:optional
	// +k8s:unionMember
	Gang *WorkloadPodGroupGangSchedulingPolicy `json:"gang,omitempty" protobuf:"bytes,2,opt,name=gang"`
}

// WorkloadPodGroupBasicSchedulingPolicy indicates standard Kubernetes
// scheduling behavior.
type WorkloadPodGroupBasicSchedulingPolicy struct {
	// Intentionally empty for now.
}

// WorkloadPodGroupGangSchedulingPolicy defines the parameters for gang
// (all-or-nothing) scheduling.
type WorkloadPodGroupGangSchedulingPolicy struct {
	// minCount is the minimum number of pods that must be scheduled
	// at the same time for the scheduler to admit the entire group.
	// This field is optional. If it is not specified, the controller
	// should inject a context-specific sane default (e.g.,
	// parallelism for a Job).
	// If set, it must be a positive integer.
	//
	// +optional
	// +k8s:optional
	// +k8s:minimum=1
	MinCount *int32 `json:"minCount,omitempty" protobuf:"varint,1,opt,name=minCount"`
}

// WorkloadPodGroupSchedulingConstraints defines leaf-level scheduling
// constraints, such as topology.
//
// ---
//
// This is a reusable building block meant to be embedded in a controller's own
// API, next to its other scheduling fields (for example, in a Job's
// spec.scheduling). It's recommended to freeze the field after creation
// (+k8s:immutable), since constraints are immutable in the compiled Workload.
// For example,
//
//	type JobSchedulingConfiguration struct {
//		// SchedulingConstraints defines scheduling constraints (e.g. topology)
//		// for the Job's pods.
//		// This field is immutable after creation.
//		// +optional
//		// +k8s:optional
//		// +k8s:immutable
//		SchedulingConstraints *schedulingv1.WorkloadPodGroupSchedulingConstraints `json:"schedulingConstraints,omitempty" protobuf:"bytes,N,opt,name=schedulingConstraints"`
//
//		// other scheduling fields
//	}
type WorkloadPodGroupSchedulingConstraints struct {
	// topology specifies desired topological placements for all pods
	// within the pod group.
	// If unset, no topology placement is requested.
	//
	// +optional
	// +k8s:optional
	// +k8s:maxItems=1
	// +listType=atomic
	// +k8s:listType=atomic
	Topology []WorkloadTopologyConstraint `json:"topology,omitempty" protobuf:"bytes,1,rep,name=topology"`
}

// WorkloadPodGroupResourceClaim references a dynamic resource claim
// that is shared across pods in the group.
//
// ---
//
// This is a reusable building block meant to be embedded in a controller's own
// API as a list, next to its other scheduling fields (for example, in a Job's
// spec.scheduling). It's recommended to freeze the whole list after creation
// (+k8s:immutable), since the list is immutable in the compiled Workload. For
// example,
//
//	type JobSchedulingConfiguration struct {
//		// ResourceClaims lists the ResourceClaims shared among the group's pods.
//		// At most 4 claims may be set, matching the limit on the resulting PodGroup.
//		// This list is immutable after creation: entries may neither be added,
//		// removed, nor modified.
//		// +optional
//		// +patchMergeKey=name
//		// +patchStrategy=merge
//		// +listType=map
//		// +listMapKey=name
//		// +k8s:optional
//		// +k8s:listType=map
//		// +k8s:listMapKey=name
//		// +k8s:maxItems=4
//		// +k8s:immutable
//		ResourceClaims []schedulingv1.WorkloadPodGroupResourceClaim `json:"resourceClaims,omitempty" protobuf:"bytes,N,rep,name=resourceClaims"`
//
//		// other scheduling fields
//	}
type WorkloadPodGroupResourceClaim struct {
	// name uniquely identifies this resource claim inside the group.
	// This field is required. It must be a DNS_LABEL.
	//
	// +required
	// +k8s:required
	// +k8s:format=k8s-short-name
	Name string `json:"name" protobuf:"bytes,1,opt,name=name"`

	// resourceClaimName is the name of a ResourceClaim object in the same
	// namespace.
	// This field is optional. If it is not specified, no resource claim
	// is used. If set, it must be a DNS subdomain.
	//
	// +optional
	// +k8s:optional
	// +k8s:unionMember
	// +k8s:format=k8s-long-name
	ResourceClaimName *string `json:"resourceClaimName,omitempty" protobuf:"bytes,2,opt,name=resourceClaimName"`

	// resourceClaimTemplateName is the name of a ResourceClaimTemplate
	// object in the same namespace.
	// This field is optional. If it is not specified, no resource claim
	// template is used. If set, it must be a DNS subdomain.
	//
	// +optional
	// +k8s:optional
	// +k8s:unionMember
	// +k8s:format=k8s-long-name
	ResourceClaimTemplateName *string `json:"resourceClaimTemplateName,omitempty" protobuf:"bytes,3,opt,name=resourceClaimTemplateName"`
}

// WorkloadPodGroupDisruptionMode defines how individual pods within a
// group can be disrupted. Exactly one mode must be set.
//
// ---
//
// This is a reusable building block meant to be embedded in a controller's own
// API, next to its other scheduling fields (for example, in a Job's
// spec.scheduling). It's recommended to freeze the field after creation
// (+k8s:immutable), since the selected mode is immutable in the compiled
// Workload. For example,
//
//	type JobSchedulingConfiguration struct {
//		// DisruptionMode defines the mode in which the Job's pods can be disrupted.
//		// One of Single, All.
//		// This field is immutable after creation: it may not be added or removed,
//		// and the selected mode may not be changed.
//		// +optional
//		// +k8s:optional
//		// +k8s:immutable
//		DisruptionMode *schedulingv1.WorkloadPodGroupDisruptionMode `json:"disruptionMode,omitempty" protobuf:"bytes,N,opt,name=disruptionMode"`
//
//		// other scheduling fields
//	}
//
// +union
type WorkloadPodGroupDisruptionMode struct {
	// single specifies that pods can be disrupted independently from each other.
	//
	// +optional
	// +k8s:optional
	// +k8s:unionMember
	Single *WorkloadPodGroupSingleDisruptionMode `json:"single,omitempty" protobuf:"bytes,1,opt,name=single"`

	// all specifies that all pods in the group must be disrupted together.
	//
	// +optional
	// +k8s:optional
	// +k8s:unionMember
	All *WorkloadPodGroupAllDisruptionMode `json:"all,omitempty" protobuf:"bytes,2,opt,name=all"`
}

// WorkloadPodGroupSingleDisruptionMode indicates that individual pods
// can be disrupted independently.
type WorkloadPodGroupSingleDisruptionMode struct {
	// Intentionally empty for now.
}

// WorkloadPodGroupAllDisruptionMode indicates that all pods in the
// group must be disrupted together.
type WorkloadPodGroupAllDisruptionMode struct {
	// Intentionally empty for now.
}

// WorkloadCompositePodGroupSchedulingPolicy defines the scheduling policy for a
// CompositePodGroup (a group of groups) managed by a workload controller.
// Exactly one policy must be set.
//
// ---
//
// This is a reusable building block meant to be embedded in a composite
// controller's own API, next to its other scheduling fields. It is the
// group-of-groups analogue of WorkloadPodGroupSchedulingPolicy. The composite
// scheduling policy is immutable, so it's recommended to freeze the whole
// field after creation (+k8s:immutable). For example,
//
//	type JobSetSchedulingConfiguration struct {
//		// SchedulingPolicy defines the scheduling policy for this JobSet.
//		// Exactly one of Basic or Gang must be set.
//		// This field is immutable after creation.
//		// +optional
//		// +k8s:optional
//		// +k8s:immutable
//		SchedulingPolicy *schedulingv1.WorkloadCompositePodGroupSchedulingPolicy `json:"schedulingPolicy,omitempty" protobuf:"bytes,N,opt,name=schedulingPolicy"`
//
//		// other scheduling fields
//	}
//
// +union
type WorkloadCompositePodGroupSchedulingPolicy struct {
	// basic specifies that the child groups of this composite group should be
	// scheduled independently.
	//
	// +optional
	// +k8s:optional
	// +k8s:unionMember
	Basic *WorkloadCompositePodGroupBasicSchedulingPolicy `json:"basic,omitempty" protobuf:"bytes,1,opt,name=basic"`

	// gang specifies that the child groups of this composite group should be
	// scheduled using all-or-nothing semantics.
	//
	// +optional
	// +k8s:optional
	// +k8s:unionMember
	Gang *WorkloadCompositePodGroupGangSchedulingPolicy `json:"gang,omitempty" protobuf:"bytes,2,opt,name=gang"`
}

// WorkloadCompositePodGroupBasicSchedulingPolicy indicates that the child groups
// of a composite group should be scheduled independently.
type WorkloadCompositePodGroupBasicSchedulingPolicy struct {
	// Intentionally empty for now.
}

// WorkloadCompositePodGroupGangSchedulingPolicy defines the parameters for gang
// scheduling of a composite group's child groups.
type WorkloadCompositePodGroupGangSchedulingPolicy struct {
	// minGroupCount is the minimum number of child groups that must be
	// schedulable at the same time for the scheduler to admit the entire
	// composite group.
	// This field is optional. If it is not specified, the controller
	// should inject a context-specific sane default (e.g., the number of
	// child groups).
	// If set, it must be a positive integer.
	//
	// +optional
	// +k8s:optional
	// +k8s:minimum=1
	MinGroupCount *int32 `json:"minGroupCount,omitempty" protobuf:"varint,1,opt,name=minGroupCount"`
}

// WorkloadCompositePodGroupSchedulingConstraints defines composite-level
// scheduling constraints, such as topology, for a CompositePodGroup (a group of
// groups) managed by a workload controller.
//
// ---
//
// This is a reusable building block meant to be embedded in a composite
// controller's own API, next to its other scheduling fields. It is the
// group-of-groups analogue of WorkloadPodGroupSchedulingConstraints. It's
// recommended to freeze the field after creation (+k8s:immutable), since
// constraints are immutable in the compiled Workload. For example,
//
//	type JobSetSchedulingConfiguration struct {
//		// SchedulingConstraints defines scheduling constraints (e.g. topology)
//		// for this JobSet.
//		// This field is immutable after creation.
//		// +optional
//		// +k8s:optional
//		// +k8s:immutable
//		SchedulingConstraints *schedulingv1.WorkloadCompositePodGroupSchedulingConstraints `json:"schedulingConstraints,omitempty" protobuf:"bytes,N,opt,name=schedulingConstraints"`
//
//		// other scheduling fields
//	}
type WorkloadCompositePodGroupSchedulingConstraints struct {
	// topology specifies desired topological placements for all child groups
	// within the composite group.
	// If unset, no topology placement is requested.
	//
	// +optional
	// +k8s:optional
	// +k8s:maxItems=1
	// +listType=atomic
	// +k8s:listType=atomic
	Topology []WorkloadTopologyConstraint `json:"topology,omitempty" protobuf:"bytes,1,rep,name=topology"`
}

// WorkloadCompositePodGroupDisruptionMode defines how the child groups of a
// composite group can be disrupted. Exactly one mode must be set.
//
// ---
//
// This is a reusable building block meant to be embedded in a composite
// controller's own API, next to its other scheduling fields. It is the
// group-of-groups analogue of WorkloadPodGroupDisruptionMode. It's recommended
// to freeze the field after creation (+k8s:immutable), since the selected mode
// is immutable in the compiled Workload. For example,
//
//	type JobSetSchedulingConfiguration struct {
//		// DisruptionMode defines the mode in which this JobSet's child groups
//		// can be disrupted.
//		// One of Single, All.
//		// This field is immutable after creation: it may not be added or removed,
//		// and the selected mode may not be changed.
//		// +optional
//		// +k8s:optional
//		// +k8s:immutable
//		DisruptionMode *schedulingv1.WorkloadCompositePodGroupDisruptionMode `json:"disruptionMode,omitempty" protobuf:"bytes,N,opt,name=disruptionMode"`
//
//		// other scheduling fields
//	}
//
// +union
type WorkloadCompositePodGroupDisruptionMode struct {
	// single specifies that child groups can be disrupted independently from
	// each other.
	//
	// +optional
	// +k8s:optional
	// +k8s:unionMember
	Single *WorkloadCompositePodGroupSingleDisruptionMode `json:"single,omitempty" protobuf:"bytes,1,opt,name=single"`

	// all specifies that all child groups must be disrupted together.
	//
	// +optional
	// +k8s:optional
	// +k8s:unionMember
	All *WorkloadCompositePodGroupAllDisruptionMode `json:"all,omitempty" protobuf:"bytes,2,opt,name=all"`
}

// WorkloadCompositePodGroupSingleDisruptionMode indicates that child groups
// can be disrupted independently.
type WorkloadCompositePodGroupSingleDisruptionMode struct {
	// Intentionally empty for now.
}

// WorkloadCompositePodGroupAllDisruptionMode indicates that all child groups
// must be disrupted together.
type WorkloadCompositePodGroupAllDisruptionMode struct {
	// Intentionally empty for now.
}
