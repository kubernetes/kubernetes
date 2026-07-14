/*
Copyright The Kubernetes Authors.

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

package v1alpha3

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Workload allows for expressing scheduling constraints that should be used
// when managing the lifecycle of workloads from the scheduling perspective,
// including scheduling, preemption, eviction and other phases.
// Workload API enablement is toggled by the GenericWorkload feature gate.
type Workload struct {
	metav1.TypeMeta `json:""`
	// metadata is the standard object metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	//
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// spec defines the desired behavior of a Workload.
	//
	// +required
	Spec WorkloadSpec `json:"spec" protobuf:"bytes,2,opt,name=spec"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// WorkloadList contains a list of Workload resources.
type WorkloadList struct {
	metav1.TypeMeta `json:""`
	// Standard list metadata.
	//
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Items is the list of Workloads.
	Items []Workload `json:"items" protobuf:"bytes,2,rep,name=items"`
}

const (
	// WorkloadMaxPodGroupTemplates is the maximum number of pod group templates per Workload.
	WorkloadMaxPodGroupTemplates = 8
	// WorkloadMaxTreeDepth is the maximum allowed depth for a tree of (composite) pod group templates in a Workload.
	WorkloadMaxTreeDepth = 4
)

// WorkloadSpec defines the desired state of a Workload.
type WorkloadSpec struct {
	// controllerRef is an optional reference to the controlling object, such as a
	// Deployment or Job. This field is intended for use by tools like CLIs
	// to provide a link back to the original workload definition.
	// This field is immutable.
	//
	// +optional
	// +k8s:optional
	// +k8s:immutable
	ControllerRef *TypedLocalObjectReference `json:"controllerRef,omitempty" protobuf:"bytes,1,opt,name=controllerRef"`

	// podGroupTemplates is the list of templates that make up the Workload.
	// The maximum number of templates is 8. Templates cannot be added or removed after the workload is created.
	// Existing templates may still be updated where their individual fields allow it.
	// Exactly one of CompositePodGroupTemplates and PodGroupTemplates must be set.
	//
	// +optional
	// +listType=map
	// +listMapKey=name
	// +k8s:optional
	// +k8s:unionMember
	// +k8s:listType=map
	// +k8s:listMapKey=name
	// +k8s:maxItems=8
	// +k8s:update=NoAddItem
	// +k8s:update=NoRemoveItem
	PodGroupTemplates []PodGroupTemplate `json:"podGroupTemplates" protobuf:"bytes,2,rep,name=podGroupTemplates"`

	// compositePodGroupTemplates is the list of CompositePodGroup templates that make up the Workload.
	// The maximum number of templates is 8. This field is immutable.
	// Exactly one of CompositePodGroupTemplates and PodGroupTemplates must be set.
	//
	// This field is used only when the CompositePodGroup feature gate is enabled.
	//
	// +featureGate=CompositePodGroup
	// +optional
	// +listType=map
	// +listMapKey=name
	// +k8s:ifDisabled("CompositePodGroup")=+k8s:forbidden
	// +k8s:ifEnabled("CompositePodGroup")=+k8s:optional
	// +k8s:unionMember
	// +k8s:listType=map
	// +k8s:listMapKey=name
	// +k8s:maxItems=8
	// +k8s:update=NoAddItem
	// +k8s:update=NoRemoveItem
	CompositePodGroupTemplates []CompositePodGroupTemplate `json:"compositePodGroupTemplates,omitempty" protobuf:"bytes,3,rep,name=compositePodGroupTemplates"`
}

// TypedLocalObjectReference allows to reference typed object inside the same namespace.
type TypedLocalObjectReference struct {
	// apiGroup is the group for the resource being referenced.
	// If APIGroup is empty, the specified Kind must be in the core API group.
	// For any other third-party types, setting APIGroup is required.
	// It must be a DNS subdomain.
	//
	// +optional
	// +k8s:optional
	// +k8s:format=k8s-long-name
	APIGroup string `json:"apiGroup,omitempty" protobuf:"bytes,1,opt,name=apiGroup"`
	// kind is the type of resource being referenced.
	// It must be a path segment name.
	//
	// +required
	// +k8s:required
	// +k8s:format=k8s-path-segment-name
	Kind string `json:"kind" protobuf:"bytes,2,opt,name=kind"`
	// name is the name of resource being referenced.
	// It must be a path segment name.
	//
	// +required
	// +k8s:required
	// +k8s:format=k8s-path-segment-name
	Name string `json:"name" protobuf:"bytes,3,opt,name=name"`
}

// MaxPodGroupResourceClaims is the maximum number of resource claims for a
// PodGroup or a Workload's PodGroupTemplate.
const MaxPodGroupResourceClaims = 4

// PodGroupTemplate represents a template for a set of pods with a scheduling policy.
type PodGroupTemplate struct {
	// name is a unique identifier for the PodGroupTemplate within the Workload.
	// It must be a DNS label. This field is immutable.
	//
	// +required
	// +k8s:required
	// +k8s:format=k8s-short-name
	Name string `json:"name" protobuf:"bytes,1,opt,name=name"`

	// schedulingPolicy defines the scheduling policy for this PodGroupTemplate.
	//
	// +required
	SchedulingPolicy PodGroupSchedulingPolicy `json:"schedulingPolicy" protobuf:"bytes,2,opt,name=schedulingPolicy"`

	// schedulingConstraints defines optional scheduling constraints (e.g. topology) for this PodGroupTemplate.
	// This field is only available when the TopologyAwareWorkloadScheduling feature gate is enabled.
	// This field is immutable.
	//
	// +featureGate=TopologyAwareWorkloadScheduling
	// +optional
	// +k8s:ifDisabled(TopologyAwareWorkloadScheduling)=+k8s:forbidden
	// +k8s:ifEnabled(TopologyAwareWorkloadScheduling)=+k8s:optional
	// +k8s:immutable
	SchedulingConstraints *PodGroupSchedulingConstraints `json:"schedulingConstraints" protobuf:"bytes,3,opt,name=schedulingConstraints"`

	// resourceClaims defines which ResourceClaims may be shared among Pods in
	// the group. Pods consume the devices allocated to a PodGroup's claim by
	// defining a claim in its own Spec.ResourceClaims that matches the
	// PodGroup's claim exactly. The claim must have the same name and refer to
	// the same ResourceClaim or ResourceClaimTemplate.
	//
	// This is an alpha-level field and requires that the
	// DRAWorkloadResourceClaims feature gate is enabled.
	//
	// This field is immutable.
	//
	// +optional
	// +patchMergeKey=name
	// +patchStrategy=merge,retainKeys
	// +listType=map
	// +listMapKey=name
	// +k8s:optional
	// +k8s:listType=map
	// +k8s:listMapKey=name
	// +k8s:maxItems=4
	// +featureGate=DRAWorkloadResourceClaims
	// +k8s:immutable
	ResourceClaims []PodGroupResourceClaim `json:"resourceClaims,omitempty" patchStrategy:"merge,retainKeys" patchMergeKey:"name" protobuf:"bytes,4,rep,name=resourceClaims"`

	// disruptionMode defines the mode in which a given PodGroup can be disrupted.
	// One of Single, All.
	// This field is immutable.
	//
	// +optional
	// +k8s:optional
	// +k8s:immutable
	DisruptionMode *DisruptionMode `json:"disruptionMode,omitempty" protobuf:"bytes,5,opt,name=disruptionMode"`

	// priorityClassName indicates the priority that should be considered when scheduling
	// a pod group created from this template.
	// This field is immutable.
	//
	// +optional
	// +k8s:optional
	// +k8s:format=k8s-long-name
	// +k8s:immutable
	PriorityClassName string `json:"priorityClassName,omitempty" protobuf:"bytes,6,opt,name=priorityClassName"`

	// priority is the value of priority of pod groups created from this template. Various
	// system components use this field to find the priority of the pod group.
	// The higher the value, the higher the priority.
	// This field is immutable.
	//
	// +optional
	// +k8s:optional
	// +k8s:maximum=1000000000 # HighestUserDefinablePriority
	// +k8s:immutable
	Priority *int32 `json:"priority,omitempty" protobuf:"varint,7,opt,name=priority"`

	// preemptionPolicy is the Policy for preempting pods/podgroups with lower priority.
	// One of Never, PreemptLowerPriority.
	// This field is immutable.
	// This field is available only when the PodGroupPreemptionPolicy feature gate is enabled.
	//
	// +featureGate=PodGroupPreemptionPolicy
	// +optional
	// +k8s:immutable
	// +k8s:ifDisabled("PodGroupPreemptionPolicy")=+k8s:forbidden
	// +k8s:ifEnabled("PodGroupPreemptionPolicy")=+k8s:optional
	PreemptionPolicy *PreemptionPolicy `json:"preemptionPolicy,omitempty" protobuf:"bytes,8,opt,name=preemptionPolicy"`
}

// CompositePodGroupTemplate represents a template for a CompositePodGroup with a scheduling policy.
type CompositePodGroupTemplate struct {
	// name is a unique identifier for the CompositePodGroupTemplate within the Workload.
	// It must be a DNS label. This field is required.
	//
	// +required
	// +k8s:required
	// +k8s:format=k8s-short-name
	Name string `json:"name" protobuf:"bytes,1,opt,name=name"`

	// schedulingPolicy defines the scheduling policy for this template.
	//
	// +required
	SchedulingPolicy CompositePodGroupSchedulingPolicy `json:"schedulingPolicy" protobuf:"bytes,2,opt,name=schedulingPolicy"`

	// priorityClassName indicates the priority that should be considered when scheduling
	// a composite pod group created from this template. If no priority class is specified,
	// admission control can set this to the global default priority class if it exists.
	// Otherwise, composite pod groups created from this template will have the priority set
	// to zero.
	// This field is immutable.
	//
	// +optional
	// +k8s:optional
	// +k8s:format=k8s-long-name
	// +k8s:immutable
	PriorityClassName string `json:"priorityClassName,omitempty" protobuf:"bytes,3,opt,name=priorityClassName"`

	// priority is the value of priority of composite pod groups created from this template.
	// Various system components use this field to find the priority of the composite pod group.
	// When Priority Admission Controller is enabled, it prevents users from setting this field.
	// The admission controller populates this field from PriorityClassName.
	// The higher the value, the higher the priority.
	// This field is immutable.
	//
	// +optional
	// +k8s:optional
	// +k8s:maximum=1000000000 # HighestUserDefinablePriority
	// +k8s:immutable
	Priority *int32 `json:"priority,omitempty" protobuf:"varint,4,opt,name=priority"`

	// podGroupTemplates is the list of templates for children PodGroups.
	// The maximum number of templates is 8. At least one entry in CompositePodGroupTemplates
	// or PodGroupTemplates must be set.
	//
	// +optional
	// +listType=map
	// +listMapKey=name
	// +k8s:optional
	// +k8s:listType=map
	// +k8s:listMapKey=name
	// +k8s:maxItems=8
	// +k8s:update=NoAddItem
	// +k8s:update=NoRemoveItem
	PodGroupTemplates []PodGroupTemplate `json:"podGroupTemplates,omitempty" protobuf:"bytes,5,rep,name=podGroupTemplates"`

	// compositePodGroupTemplates is the list of templates for children CompositePodGroups.
	// The maximum number of templates is 8. At least one entry in CompositePodGroupTemplates
	// or PodGroupTemplates must be set.
	//
	// +optional
	// +listType=map
	// +listMapKey=name
	// +k8s:optional
	// +k8s:listType=map
	// +k8s:listMapKey=name
	// +k8s:maxItems=8
	// +k8s:update=NoAddItem
	// +k8s:update=NoRemoveItem
	CompositePodGroupTemplates []CompositePodGroupTemplate `json:"compositePodGroupTemplates,omitempty" protobuf:"bytes,6,rep,name=compositePodGroupTemplates"`

	// schedulingConstraints defines optional scheduling constraints (e.g. topology) for this CompositePodGroupTemplate.
	// This field is immutable.
	//
	// +optional
	// +k8s:optional
	// +k8s:immutable
	SchedulingConstraints *CompositePodGroupSchedulingConstraints `json:"schedulingConstraints,omitempty" protobuf:"bytes,7,opt,name=schedulingConstraints"`
}

// PodGroupSchedulingPolicy defines the scheduling configuration for a PodGroup.
// Exactly one policy must be set. The policy is chosen at creation time by setting either the
// Basic or Gang field. The PodGroup may not change policy after creation.
// Fields within chosen policy may be updated after creation when their individual fields allow it.
//
// +union
type PodGroupSchedulingPolicy struct {
	// basic specifies that the pods in this group should be scheduled using
	// standard Kubernetes scheduling behavior. Setting this field at group creation time
	// opts this group to basic scheduling; this field cannot be changed afterward.
	//
	// +optional
	// +k8s:optional
	// +k8s:unionMember
	// +k8s:immutable
	Basic *BasicSchedulingPolicy `json:"basic,omitempty" protobuf:"bytes,1,opt,name=basic"`

	// gang specifies that the pods in this group should be scheduled using
	// all-or-nothing semantics. Setting this field at group creation time
	// opts this group to gang scheduling; this field cannot be set or unset afterward.
	// The minCount field within Gang scheduling policy remains mutable after group creation.
	//
	// +optional
	// +k8s:optional
	// +k8s:unionMember
	// +k8s:update=NoSet
	// +k8s:update=NoUnset
	Gang *GangSchedulingPolicy `json:"gang,omitempty" protobuf:"bytes,2,opt,name=gang"`
}

// BasicSchedulingPolicy indicates that standard Kubernetes
// scheduling behavior should be used.
type BasicSchedulingPolicy struct {
	// This is intentionally empty. Its presence indicates that the basic
	// scheduling policy should be applied. In the future, new fields may appear,
	// describing such constraints on a pod group level without "all or nothing"
	// (gang) scheduling.
}

// GangSchedulingPolicy defines the parameters for gang scheduling.
type GangSchedulingPolicy struct {
	// minCount is the minimum number of pods that must be schedulable or scheduled
	// at the same time for the scheduler to admit the entire group.
	// It must be a positive integer. This field is mutable to support workload scaling.
	//
	// Note that the scheduler operates on an eventually consistent model. Updates
	// to minCount may not be immediately reflected in scheduling decisions due to
	// propagation delays. If minCount is updated while a scheduling cycle is in
	// progress for that group, the new value may not take effect until the next
	// cycle. Moreover, minCount is only enforced during scheduling, meaning that
	// modifications to this field do not affect already-scheduled pods, applying
	// only to those evaluated in future cycles.
	//
	// +required
	// +k8s:required
	// +k8s:minimum=1
	MinCount int32 `json:"minCount" protobuf:"varint,1,opt,name=minCount"`
}

// PodGroupResourceClaim references exactly one ResourceClaim, either directly
// or by naming a ResourceClaimTemplate which is then turned into a ResourceClaim
// for the PodGroup.
//
// It adds a name to it that uniquely identifies the ResourceClaim inside the PodGroup.
// Pods that need access to the ResourceClaim define a matching reference in its
// own Spec.ResourceClaims. The Pod's claim must match all fields of the
// PodGroup's claim exactly.
type PodGroupResourceClaim struct {
	// name uniquely identifies this resource claim inside the PodGroup.
	// This must be a DNS_LABEL.
	//
	// +required
	// +k8s:required
	// +k8s:format=k8s-short-name
	Name string `json:"name" protobuf:"bytes,1,opt,name=name"`

	// resourceClaimName is the name of a ResourceClaim object in the same
	// namespace as this PodGroup. The ResourceClaim will be reserved for the
	// PodGroup instead of its individual pods.
	//
	// Exactly one of ResourceClaimName and ResourceClaimTemplateName must
	// be set.
	//
	// +optional
	// +k8s:optional
	// +k8s:unionMember
	// +k8s:format=k8s-long-name
	ResourceClaimName *string `json:"resourceClaimName,omitempty" protobuf:"bytes,2,opt,name=resourceClaimName"`

	// resourceClaimTemplateName is the name of a ResourceClaimTemplate
	// object in the same namespace as this PodGroup.
	//
	// The template will be used to create a new ResourceClaim, which will
	// be bound to this PodGroup. When this PodGroup is deleted, the ResourceClaim
	// will also be deleted. The PodGroup name and resource name, along with a
	// generated component, will be used to form a unique name for the
	// ResourceClaim, which will be recorded in podgroup.status.resourceClaimStatuses.
	//
	// This field is immutable and no changes will be made to the
	// corresponding ResourceClaim by the control plane after creating the
	// ResourceClaim.
	//
	// Exactly one of ResourceClaimName and ResourceClaimTemplateName must
	// be set.
	//
	// +optional
	// +k8s:optional
	// +k8s:unionMember
	// +k8s:format=k8s-long-name
	ResourceClaimTemplateName *string `json:"resourceClaimTemplateName,omitempty" protobuf:"bytes,3,opt,name=resourceClaimTemplateName"`
}

// DisruptionMode defines how individual entities within a group can be disrupted.
// Exactly one mode can be set.
//
// +union
type DisruptionMode struct {
	// single specifies that children can be disrupted independently from each other.
	//
	// +optional
	// +k8s:optional
	// +k8s:unionMember
	Single *SingleDisruptionMode `json:"single,omitempty" protobuf:"bytes,1,opt,name=single"`

	// all specifies that all children can only be disrupted together.
	//
	// +optional
	// +k8s:optional
	// +k8s:unionMember
	All *AllDisruptionMode `json:"all,omitempty" protobuf:"bytes,2,opt,name=all"`
}

// SingleDisruptionMode specifies that children can be disrupted independently.
type SingleDisruptionMode struct {
	// Intentionally empty now.
}

// AllDisruptionMode specifies that children can only be disrupted together.
type AllDisruptionMode struct {
	// Intentionally empty now.
}

// PreemptionPolicy describes a policy for if/when to preempt a pod.
// +enum
// +k8s:enum
type PreemptionPolicy string

const (
	// PreemptLowerPriority means that pod can preempt other pods with lower priority.
	PreemptLowerPriority PreemptionPolicy = "PreemptLowerPriority"
	// PreemptNever means that pod never preempts other pods with lower priority.
	PreemptNever PreemptionPolicy = "Never"
)

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:supportsSubresource="/status"

// PodGroup represents a runtime instance of pods grouped together.
// PodGroups are created by workload controllers (Job, LWS, JobSet, etc...) from
// Workload.podGroupTemplates.
// PodGroup API enablement is toggled by the GenericWorkload feature gate.
type PodGroup struct {
	metav1.TypeMeta `json:""`
	// metadata is the standard object metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	//
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// spec defines the desired state of the PodGroup.
	//
	// +required
	Spec PodGroupSpec `json:"spec" protobuf:"bytes,2,opt,name=spec"`

	// status represents the current observed state of the PodGroup.
	//
	// +optional
	Status PodGroupStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// PodGroupList contains a list of PodGroup resources.
type PodGroupList struct {
	metav1.TypeMeta `json:""`
	// Standard list metadata.
	//
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Items is the list of PodGroups.
	Items []PodGroup `json:"items" protobuf:"bytes,2,rep,name=items"`
}

// PodGroupSpec defines the desired state of a PodGroup.
type PodGroupSpec struct {
	// parentCompositePodGroupName contains the name of the parent composite pod group
	// within the same namespace as this pod group.
	// If it's nil, then this pod group is a root of a workload's hierarchy.
	// This field is used only when the CompositePodGroup feature gate is enabled.
	// This field is immutable.
	//
	// +featureGate=CompositePodGroup
	// +optional
	// +k8s:ifDisabled(CompositePodGroup)=+k8s:forbidden
	// +k8s:ifEnabled(CompositePodGroup)=+k8s:optional
	// +k8s:immutable
	// +k8s:format=k8s-long-name
	// +k8s:dependentRequired("workloadRef")
	ParentCompositePodGroupName *string `json:"parentCompositePodGroupName,omitempty" protobuf:"bytes,1,opt,name=parentCompositePodGroupName"`

	// workloadRef references an optional PodGroup template within the Workload
	// object that was used to create the PodGroup.
	// This field is immutable.
	//
	// +optional
	// +k8s:optional
	// +k8s:immutable
	WorkloadRef *WorkloadReference `json:"workloadRef,omitempty" protobuf:"bytes,2,opt,name=workloadRef"`

	// schedulingPolicy defines the scheduling policy for this instance of the PodGroup.
	// Controllers are expected to fill this field by copying it from a PodGroupTemplate.
	//
	// +required
	SchedulingPolicy PodGroupSchedulingPolicy `json:"schedulingPolicy" protobuf:"bytes,3,opt,name=schedulingPolicy"`

	// schedulingConstraints defines optional scheduling constraints (e.g. topology) for this PodGroup.
	// Controllers are expected to fill this field by copying it from a PodGroupTemplate.
	// This field is immutable.
	// This field is only available when the TopologyAwareWorkloadScheduling feature gate is enabled.
	//
	// +featureGate=TopologyAwareWorkloadScheduling
	// +optional
	// +k8s:ifDisabled(TopologyAwareWorkloadScheduling)=+k8s:forbidden
	// +k8s:ifEnabled(TopologyAwareWorkloadScheduling)=+k8s:optional
	// +k8s:ifEnabled(TopologyAwareWorkloadScheduling)=+k8s:immutable
	SchedulingConstraints *PodGroupSchedulingConstraints `json:"schedulingConstraints,omitempty" protobuf:"bytes,4,opt,name=schedulingConstraints"`

	// resourceClaims defines which ResourceClaims may be shared among Pods in
	// the group. Pods consume the devices allocated to a PodGroup's claim by
	// defining a claim in its own Spec.ResourceClaims that matches the
	// PodGroup's claim exactly. The claim must have the same name and refer to
	// the same ResourceClaim or ResourceClaimTemplate.
	//
	// This is an alpha-level field and requires that the
	// DRAWorkloadResourceClaims feature gate is enabled.
	//
	// This field is immutable.
	//
	// +optional
	// +patchMergeKey=name
	// +patchStrategy=merge,retainKeys
	// +listType=map
	// +listMapKey=name
	// +k8s:optional
	// +k8s:listType=map
	// +k8s:listMapKey=name
	// +k8s:maxItems=4
	// +k8s:immutable
	// +featureGate=DRAWorkloadResourceClaims
	ResourceClaims []PodGroupResourceClaim `json:"resourceClaims,omitempty" patchStrategy:"merge,retainKeys" patchMergeKey:"name" protobuf:"bytes,5,rep,name=resourceClaims"`

	// disruptionMode defines the mode in which a given PodGroup can be disrupted.
	// Controllers are expected to fill this field by copying it from a PodGroupTemplate.
	// One of Single, All. Defaults to Single if unset.
	// This field is immutable.
	//
	// +default={"single": {}}
	// +optional
	// +k8s:optional
	// +k8s:immutable
	DisruptionMode *DisruptionMode `json:"disruptionMode,omitempty" protobuf:"bytes,6,opt,name=disruptionMode"`

	// priorityClassName defines the priority that should be considered when scheduling this pod group.
	// Controllers are expected to fill this field by copying it from a PodGroupTemplate.
	// Otherwise, it is validated and resolved similarly to the PriorityClassName on PodGroupTemplate
	// (i.e. if no priority class is specified, admission control can set this to the global default
	// priority class if it exists. Otherwise, the pod group's priority will be zero).
	// This field is immutable.
	//
	// +optional
	// +k8s:optional
	// +k8s:immutable
	// +k8s:format=k8s-long-name
	PriorityClassName string `json:"priorityClassName,omitempty" protobuf:"bytes,7,opt,name=priorityClassName"`

	// priority is the value of priority of this pod group. Various system components
	// use this field to find the priority of the pod group. When Priority Admission
	// Controller is enabled, it prevents users from setting this field. The admission
	// controller populates this field from PriorityClassName.
	// The higher the value, the higher the priority.
	// This field is immutable.
	//
	// +optional
	// +k8s:optional
	// +k8s:immutable
	// +k8s:maximum=1000000000 # HighestUserDefinablePriority
	Priority *int32 `json:"priority,omitempty" protobuf:"varint,8,opt,name=priority"`

	// preemptionPolicy is the Policy for preempting pods/podgroups with lower priority.
	// One of Never, PreemptLowerPriority. Defaults to PreemptLowerPriority if unset.
	// When Priority Admission Controller is enabled, it populates this field from PriorityClassName,
	// and defaults to PreemptLowerPriority if value is unset in PriorityClass.
	// This field is immutable.
	// This field is available only when the PodGroupPreemptionPolicy feature gate is enabled.
	//
	// +featureGate=PodGroupPreemptionPolicy
	// +optional
	// +k8s:immutable
	// +k8s:ifDisabled("PodGroupPreemptionPolicy")=+k8s:forbidden
	// +k8s:ifEnabled("PodGroupPreemptionPolicy")=+k8s:optional
	PreemptionPolicy *PreemptionPolicy `json:"preemptionPolicy,omitempty" protobuf:"bytes,9,opt,name=preemptionPolicy"`
}

// PodGroupStatus represents information about the status of a pod group.
type PodGroupStatus struct {
	// conditions represent the latest observations of the PodGroup's state.
	//
	// Known condition types:
	// - "PodGroupInitiallyScheduled": Indicates whether the scheduling requirement has been satisfied.
	// Once this condition transitions to True, it serves as a terminal state and will never revert to False,
	// even if pods are subsequently evicted and group constraints are no longer met.
	// - "DisruptionTarget": Indicates whether the PodGroup is about to be terminated
	//   due to disruption such as preemption.
	//
	// Known reasons for the PodGroupInitiallyScheduled condition:
	// - "Unschedulable": The PodGroup cannot be scheduled due to resource constraints,
	//   affinity/anti-affinity rules, or insufficient capacity for the gang.
	// - "SchedulerError": The PodGroup cannot be scheduled due to some internal error
	//   that happened during scheduling, for example due to nodeAffinity parsing errors.
	//
	// Known reasons for the DisruptionTarget condition:
	// - "PreemptionByScheduler": The PodGroup was preempted by the scheduler to make room for
	//   higher-priority PodGroups or Pods.
	//
	// +optional
	// +patchMergeKey=type
	// +patchStrategy=merge
	// +listType=map
	// +listMapKey=type
	// +k8s:alpha(since: "1.37")=+k8s:optional
	// +k8s:alpha(since: "1.37")=+k8s:listType=map
	// +k8s:alpha(since: "1.37")=+k8s:listMapKey=type
	Conditions []metav1.Condition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type" protobuf:"bytes,1,rep,name=conditions"`

	// resourceClaimStatuses is status of resource claims.
	// +optional
	// +patchMergeKey=name
	// +patchStrategy=merge,retainKeys
	// +listType=map
	// +listMapKey=name
	// +k8s:optional
	// +k8s:listType=map
	// +k8s:listMapKey=name
	// +k8s:maxItems=4
	// +featureGate=DRAWorkloadResourceClaims
	ResourceClaimStatuses []PodGroupResourceClaimStatus `json:"resourceClaimStatuses,omitempty" patchStrategy:"merge,retainKeys" patchMergeKey:"name" protobuf:"bytes,2,rep,name=resourceClaimStatuses"`
}

// Well-known condition types for PodGroups.
const (
	// PodGroupInitiallyScheduled represents status of the scheduling process for this PodGroup till first success.
	PodGroupInitiallyScheduled string = "PodGroupInitiallyScheduled"
	// DisruptionTarget indicates the PodGroup is about to be terminated due to disruption
	// such as preemption.
	DisruptionTarget string = "DisruptionTarget"
)

// Well-known condition reasons for PodGroups.
const (
	// Unschedulable reason in the PodGroupInitiallyScheduled condition indicates that the PodGroup cannot be scheduled
	// due to resource constraints, affinity/anti-affinity rules, or insufficient capacity for the PodGroup.
	PodGroupReasonUnschedulable string = "Unschedulable"
	// SchedulerError reason in the PodGroupInitiallyScheduled condition means that some internal error happens
	// during scheduling, for example due to nodeAffinity parsing errors.
	PodGroupReasonSchedulerError string = "SchedulerError"
	// PreemptionByScheduler reason in the DisruptionTarget condition indicates the PodGroup was preempted
	// to make room for higher-priority PodGroups or Pods.
	PodGroupReasonPreemptionByScheduler string = "PreemptionByScheduler"
)

// PodGroupResourceClaimStatus is stored in the PodGroupStatus for each
// PodGroupResourceClaim which references a ResourceClaimTemplate. It stores the
// generated name for the corresponding ResourceClaim.
type PodGroupResourceClaimStatus struct {
	// name uniquely identifies this resource claim inside the PodGroup. This
	// must match the name of an entry in podgroup.spec.resourceClaims, which
	// implies that the string must be a DNS_LABEL.
	//
	// +required
	Name string `json:"name" protobuf:"bytes,1,name=name"`

	// resourceClaimName is the name of the ResourceClaim that was generated for
	// the PodGroup in the namespace of the PodGroup. If this is unset, then
	// generating a ResourceClaim was not necessary. The
	// podgroup.spec.resourceClaims entry can be ignored in this case.
	//
	// +optional
	// +k8s:optional
	// +k8s:format=k8s-long-name
	ResourceClaimName *string `json:"resourceClaimName,omitempty" protobuf:"bytes,2,opt,name=resourceClaimName"`
}

// WorkloadReference references the Workload object together with the template
// that was used to create a particular PodGroup.
type WorkloadReference struct {
	// workloadName is the name of the Workload object that contains a template
	// that was used when creating a pod group. It must
	// be a DNS name.
	// This field is required.
	//
	// +required
	// +k8s:required
	// +k8s:format=k8s-long-name
	WorkloadName string `json:"workloadName" protobuf:"bytes,1,opt,name=workloadName"`

	// templateName is the name of a template within the Workload object that
	// was used to create a pod group. It must be a DNS label.
	// This field is required.
	//
	// +required
	// +k8s:required
	// +k8s:format=k8s-short-name
	TemplateName string `json:"templateName" protobuf:"bytes,2,opt,name=templateName"`
}

// PodGroupSchedulingConstraints defines scheduling constraints (e.g. topology) for a PodGroup.
type PodGroupSchedulingConstraints struct {
	// topology defines the topology constraints for the pod group.
	// Currently only a single topology constraint can be specified. This may change in the future.
	//
	// +optional
	// +k8s:optional
	// +k8s:maxItems=1
	// +listType=atomic
	// +k8s:listType=atomic
	Topology []TopologyConstraint `json:"topology,omitempty" protobuf:"bytes,1,rep,name=topology"`
}

// TopologyConstraint defines a topology constraint for a PodGroup.
type TopologyConstraint struct {
	// key specifies the key of the node label representing the topology domain.
	// All pods within the PodGroup must be colocated within the same domain instance.
	// Different PodGroups can land on different domain instances even if they derive from the same PodGroupTemplate.
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
//		SchedulingPolicy *schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy `json:"schedulingPolicy,omitempty" protobuf:"bytes,N,opt,name=schedulingPolicy"`
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
//		SchedulingConstraints *schedulingv1alpha3.WorkloadPodGroupSchedulingConstraints `json:"schedulingConstraints,omitempty" protobuf:"bytes,N,opt,name=schedulingConstraints"`
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
	Topology []TopologyConstraint `json:"topology,omitempty" protobuf:"bytes,1,rep,name=topology"`
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
//		ResourceClaims []schedulingv1alpha3.WorkloadPodGroupResourceClaim `json:"resourceClaims,omitempty" protobuf:"bytes,N,rep,name=resourceClaims"`
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
//		DisruptionMode *schedulingv1alpha3.WorkloadPodGroupDisruptionMode `json:"disruptionMode,omitempty" protobuf:"bytes,N,opt,name=disruptionMode"`
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

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:supportsSubresource="/status"

// CompositePodGroup represents a runtime instance of pod groups grouped together.
// CompositePodGroups are created by workload controllers (LWS, JobSet, etc...) from
// Workload.compositePodGroupTemplates.
// CompositePodGroup API enablement is toggled by the CompositePodGroup feature gate.
type CompositePodGroup struct {
	metav1.TypeMeta `json:""`

	// metadata is the standard object metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	//
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// spec defines the desired state of the CompositePodGroup.
	//
	// +required
	Spec CompositePodGroupSpec `json:"spec" protobuf:"bytes,2,opt,name=spec"`

	// status represents the current observed state of the CompositePodGroup.
	//
	// +optional
	Status CompositePodGroupStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// CompositePodGroupList contains a list of CompositePodGroup resources.
type CompositePodGroupList struct {
	metav1.TypeMeta `json:""`
	// Standard list metadata.
	//
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Items is the list of CompositePodGroups.
	Items []CompositePodGroup `json:"items" protobuf:"bytes,2,rep,name=items"`
}

// CompositePodGroupSpec defines the desired state of CompositePodGroup.
type CompositePodGroupSpec struct {
	// parentCompositePodGroupName contains the name of the parent composite pod group
	// within the same namespace as this composite pod group. It must be a DNS name.
	// If it's nil, then this composite pod group is a root of a workload's hierarchy.
	// This field is immutable.
	//
	// +optional
	// +k8s:optional
	// +k8s:immutable
	// +k8s:format=k8s-long-name
	ParentCompositePodGroupName *string `json:"parentCompositePodGroupName,omitempty" protobuf:"bytes,1,opt,name=parentCompositePodGroupName"`

	// workloadRef references an optional CompositePodGroup template within the
	// Workload object that was used to create the CompositePodGroup.
	// This field is required.
	// This field is immutable.
	//
	// +required
	// +k8s:required
	// +k8s:immutable
	WorkloadRef *WorkloadReference `json:"workloadRef" protobuf:"bytes,2,opt,name=workloadRef"`

	// schedulingPolicy defines the scheduling policy for this instance of the CompositePodGroup.
	// Controllers are expected to fill this field by copying it from a CompositePodGroupTemplate.
	// This field is immutable.
	//
	// +required
	// +k8s:immutable
	SchedulingPolicy CompositePodGroupSchedulingPolicy `json:"schedulingPolicy" protobuf:"bytes,3,opt,name=schedulingPolicy"`

	// priorityClassName defines the priority that should be considered when scheduling this CompositePodGroup.
	// Controllers are expected to fill this field by copying it from a CompositePodGroupTemplate.
	// If left unspecified, it is validated and resolved similarly to the PriorityClassName field in Pods
	// (i.e. if no priority class is specified, admission control can set this to the global default
	// priority class if it exists. Otherwise, the composite pod group's priority will be zero).
	// This field is immutable.
	//
	// +optional
	// +k8s:optional
	// +k8s:format=k8s-long-name
	// +k8s:immutable
	PriorityClassName string `json:"priorityClassName,omitempty" protobuf:"bytes,4,opt,name=priorityClassName"`

	// priority is the value of priority of this composite pod group. Various system components
	// use this field to find the priority of the composite pod group. When Priority Admission
	// Controller is enabled, it prevents users from setting this field. The admission
	// controller populates this field from PriorityClassName.
	// The higher the value, the higher the priority.
	// This field is immutable.
	//
	// +optional
	// +k8s:optional
	// +k8s:immutable
	// +k8s:maximum=1000000000 # HighestUserDefinablePriority
	Priority *int32 `json:"priority,omitempty" protobuf:"varint,5,opt,name=priority"`

	// schedulingConstraints defines optional scheduling constraints (e.g. topology) for this CompositePodGroup.
	// Controllers are expected to fill this field by copying it from a CompositePodGroupTemplate.
	// This field is immutable.
	//
	// +optional
	// +k8s:optional
	// +k8s:immutable
	SchedulingConstraints *CompositePodGroupSchedulingConstraints `json:"schedulingConstraints,omitempty" protobuf:"bytes,6,opt,name=schedulingConstraints"`
}

// CompositePodGroupSchedulingPolicy defines the scheduling configuration for a CompositePodGroup.
// Exactly one policy must be set.
//
// +union
type CompositePodGroupSchedulingPolicy struct {
	// basic specifies that the groups of this composite group should be scheduled independently.
	// This field is immutable.
	//
	// +optional
	// +k8s:optional
	// +k8s:immutable
	// +k8s:unionMember
	Basic *CompositeBasicSchedulingPolicy `json:"basic,omitempty" protobuf:"bytes,1,opt,name=basic"`

	// gang specifies that the groups of this composite group should be scheduled using
	// all-or-nothing semantics.
	//
	// +optional
	// +k8s:optional
	// +k8s:unionMember
	// +k8s:update=NoSet
	// +k8s:update=NoUnset
	Gang *CompositeGangSchedulingPolicy `json:"gang,omitempty" protobuf:"bytes,2,opt,name=gang"`
}

// CompositeBasicSchedulingPolicy indicates that the groups belonging to the composite group
// should be scheduled independently.
type CompositeBasicSchedulingPolicy struct {
	// This is intentionally empty. Its presence indicates that the basic group
	// scheduling policy should be applied. In the future, new fields may appear,
	// describing such constraints on a composite pod group level without "all or
	// nothing" (gang) scheduling.
}

// CompositeGangSchedulingPolicy indicates that the groups belonging to the composite group
// should be scheduled using all-or-nothing semantics.
type CompositeGangSchedulingPolicy struct {
	// minGroupCount is the minimum number of child groups that must be schedulable
	// or scheduled at the same time for the scheduler to admit the entire group.
	// It must be a positive integer.
	//
	// +required
	// +k8s:required
	// +k8s:minimum=1
	MinGroupCount int32 `json:"minGroupCount" protobuf:"varint,1,req,name=minGroupCount"`
}

// CompositePodGroupStatus represents information about the status of a composite pod group.
type CompositePodGroupStatus struct {
	// conditions represent the latest observations of the CompositePodGroup's state.
	//
	// Known condition types:
	// - "CompositePodGroupInitiallyScheduled": Indicates whether the overall scheduling requirement
	//   for the subtree under this CompositePodGroup has been satisfied. Once this condition
	//   transitions to True, it serves as a terminal state and will never revert to False,
	//   even if pods are subsequently deleted and group constraints are no longer met.
	// - "DisruptionTarget": Indicates whether the CompositePodGroup is about to be terminated
	//   due to disruption such as preemption.
	//
	// Known reasons for the CompositePodGroupInitiallyScheduled condition:
	// - "Unschedulable": The CompositePodGroup's subtree could not be placed due to resource constraints,
	//   affinity/anti-affinity, or topological constraints.
	// - "SchedulerError": The CompositePodGroup cannot be scheduled due to some internal error
	//   that occurred during scheduling.
	// - "Invalid": Set to True when kube-scheduler detects an invalid group layout during
	//   runtime validation. The `message` field details the specific layout violation (such as
	//   a detected cycle, exceeding the maximum depth of 4, or referencing multiple distinct Workloads).
	//
	// Known reasons for the DisruptionTarget condition:
	// - "PreemptionByScheduler": The CompositePodGroup was targeted by the scheduler's preemption loop
	//   to free up capacity for higher-priority preemptors.
	//
	// +optional
	// +patchMergeKey=type
	// +patchStrategy=merge
	// +listType=map
	// +listMapKey=type
	// +k8s:alpha(since: "1.37")=+k8s:optional
	// +k8s:alpha(since: "1.37")=+k8s:listType=map
	// +k8s:alpha(since: "1.37")=+k8s:listMapKey=type
	Conditions []metav1.Condition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type" protobuf:"bytes,1,rep,name=conditions"`
}

// CompositePodGroupSchedulingConstraints defines scheduling constraints (e.g. topology) for a CompositePodGroup.
type CompositePodGroupSchedulingConstraints struct {
	// topology defines the topology constraints for the composite pod group.
	// Currently only a single topology constraint can be specified. This may change in the future.
	//
	// +optional
	// +k8s:optional
	// +k8s:maxItems=1
	// +listType=atomic
	// +k8s:listType=atomic
	Topology []TopologyConstraint `json:"topology,omitempty" protobuf:"bytes,1,rep,name=topology"`
}
