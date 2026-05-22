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

package v1alpha2

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
	// Standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	//
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Spec defines the desired behavior of a Workload.
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

// WorkloadMaxPodGroupTemplates is the maximum number of pod group templates per Workload.
const WorkloadMaxPodGroupTemplates = 8

// WorkloadSpec defines the desired state of a Workload.
type WorkloadSpec struct {
	// ControllerRef is an optional reference to the controlling object, such as a
	// Deployment or Job. This field is intended for use by tools like CLIs
	// to provide a link back to the original workload definition.
	// This field is immutable.
	//
	// +optional
	// +k8s:optional
	// +k8s:immutable
	ControllerRef *TypedLocalObjectReference `json:"controllerRef,omitempty" protobuf:"bytes,1,opt,name=controllerRef"`

	// PodGroupTemplates is the list of templates that make up the Workload.
	// The maximum number of templates is 8. This field is immutable.
	//
	// +required
	// +listType=map
	// +listMapKey=name
	// +k8s:required
	// +k8s:listType=map
	// +k8s:listMapKey=name
	// +k8s:maxItems=8
	// +k8s:immutable
	PodGroupTemplates []PodGroupTemplate `json:"podGroupTemplates" protobuf:"bytes,2,rep,name=podGroupTemplates"`
}

// TypedLocalObjectReference allows to reference typed object inside the same namespace.
type TypedLocalObjectReference struct {
	// APIGroup is the group for the resource being referenced.
	// If APIGroup is empty, the specified Kind must be in the core API group.
	// For any other third-party types, setting APIGroup is required.
	// It must be a DNS subdomain.
	//
	// +optional
	// +k8s:optional
	// +k8s:format=k8s-long-name
	APIGroup string `json:"apiGroup,omitempty" protobuf:"bytes,1,opt,name=apiGroup"`
	// Kind is the type of resource being referenced.
	// It must be a path segment name.
	//
	// +required
	// +k8s:required
	// +k8s:format=k8s-path-segment-name
	Kind string `json:"kind" protobuf:"bytes,2,opt,name=kind"`
	// Name is the name of resource being referenced.
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
	// Name is a unique identifier for the PodGroupTemplate within the Workload.
	// It must be a DNS label. This field is immutable.
	//
	// +required
	// +k8s:required
	// +k8s:format=k8s-short-name
	Name string `json:"name" protobuf:"bytes,1,opt,name=name"`

	// SchedulingPolicy defines the scheduling policy for this PodGroupTemplate.
	//
	// +required
	SchedulingPolicy PodGroupSchedulingPolicy `json:"schedulingPolicy" protobuf:"bytes,2,opt,name=schedulingPolicy"`

	// SchedulingConstraints defines optional scheduling constraints (e.g. topology) for this PodGroupTemplate.
	// This field is only available when the TopologyAwareWorkloadScheduling feature gate is enabled.
	//
	// +featureGate=TopologyAwareWorkloadScheduling
	// +optional
	// +k8s:ifDisabled(TopologyAwareWorkloadScheduling)=+k8s:forbidden
	// +k8s:ifEnabled(TopologyAwareWorkloadScheduling)=+k8s:optional
	SchedulingConstraints *PodGroupSchedulingConstraints `json:"schedulingConstraints" protobuf:"bytes,3,opt,name=schedulingConstraints"`

	// ResourceClaims defines which ResourceClaims may be shared among Pods in
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
	ResourceClaims []PodGroupResourceClaim `json:"resourceClaims,omitempty" patchStrategy:"merge,retainKeys" patchMergeKey:"name" protobuf:"bytes,4,rep,name=resourceClaims"`

	// DisruptionMode defines the mode in which a given PodGroup can be disrupted.
	// One of Pod, PodGroup.
	// This field is available only when the WorkloadAwarePreemption feature gate
	// is enabled.
	//
	// +featureGate=WorkloadAwarePreemption
	// +optional
	// +k8s:ifDisabled("WorkloadAwarePreemption")=+k8s:forbidden
	// +k8s:ifEnabled("WorkloadAwarePreemption")=+k8s:optional
	DisruptionMode *DisruptionMode `json:"disruptionMode,omitempty" protobuf:"bytes,5,opt,name=disruptionMode,casttype=DisruptionMode"`

	// PriorityClassName indicates the priority that should be considered when scheduling
	// a pod group created from this template. If no priority class is specified, admission
	// control can set this to the global default priority class if it exists. Otherwise,
	// pod groups created from this template will have the priority set to zero.
	// This field is available only when the WorkloadAwarePreemption feature gate
	// is enabled.
	//
	// +featureGate=WorkloadAwarePreemption
	// +optional
	// +k8s:ifDisabled("WorkloadAwarePreemption")=+k8s:forbidden
	// +k8s:ifEnabled("WorkloadAwarePreemption")=+k8s:optional
	// +k8s:ifEnabled("WorkloadAwarePreemption")=+k8s:format=k8s-long-name
	PriorityClassName string `json:"priorityClassName,omitempty" protobuf:"bytes,6,opt,name=priorityClassName"`

	// Priority is the value of priority of pod groups created from this template. Various
	// system components use this field to find the priority of the pod group. When
	// Priority Admission Controller is enabled, it prevents users from setting this field.
	// The admission controller populates this field from PriorityClassName.
	// The higher the value, the higher the priority.
	// This field is available only when the WorkloadAwarePreemption feature gate
	// is enabled.
	//
	// +featureGate=WorkloadAwarePreemption
	// +optional
	// +k8s:ifDisabled("WorkloadAwarePreemption")=+k8s:forbidden
	// +k8s:ifEnabled("WorkloadAwarePreemption")=+k8s:optional
	// +k8s:ifEnabled("WorkloadAwarePreemption")=+k8s:maximum=1000000000 # HighestUserDefinablePriority
	Priority *int32 `json:"priority,omitempty" protobuf:"varint,7,opt,name=priority"`
}

// PodGroupSchedulingPolicy defines the scheduling configuration for a PodGroup.
// Exactly one policy must be set.
// +union
type PodGroupSchedulingPolicy struct {
	// Basic specifies that the pods in this group should be scheduled using
	// standard Kubernetes scheduling behavior.
	//
	// +optional
	// +k8s:optional
	// +k8s:unionMember
	Basic *BasicSchedulingPolicy `json:"basic,omitempty" protobuf:"bytes,1,opt,name=basic"`

	// Gang specifies that the pods in this group should be scheduled using
	// all-or-nothing semantics.
	//
	// +optional
	// +k8s:optional
	// +k8s:unionMember
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
	// MinCount is the minimum number of pods that must be schedulable or scheduled
	// at the same time for the scheduler to admit the entire group.
	// It must be a positive integer.
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
	// Name uniquely identifies this resource claim inside the PodGroup.
	// This must be a DNS_LABEL.
	//
	// +required
	// +k8s:required
	// +k8s:format=k8s-short-name
	Name string `json:"name" protobuf:"bytes,1,opt,name=name"`

	// ResourceClaimName is the name of a ResourceClaim object in the same
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

	// ResourceClaimTemplateName is the name of a ResourceClaimTemplate
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

// DisruptionMode describes the mode in which a PodGroup can be disrupted (e.g. preempted).
// +enum
// +k8s:enum
type DisruptionMode string

const (
	// DisruptionModePod means that individual pods can be disrupted or preempted independently.
	// It doesn't depend on exact set of pods currently running in this PodGroup.
	DisruptionModePod DisruptionMode = "Pod"
	// DisruptionModePodGroup means that the whole PodGroup needs to be disrupted
	// or preempted together.
	DisruptionModePodGroup DisruptionMode = "PodGroup"
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
	// Standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	//
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Spec defines the desired state of the PodGroup.
	//
	// +required
	Spec PodGroupSpec `json:"spec" protobuf:"bytes,2,opt,name=spec"`

	// Status represents the current observed state of the PodGroup.
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
	// PodGroupTemplateRef references an optional PodGroup template within other object
	// (e.g. Workload) that was used to create the PodGroup. This field is immutable.
	//
	// +optional
	// +k8s:optional
	// +k8s:immutable
	PodGroupTemplateRef *PodGroupTemplateReference `json:"podGroupTemplateRef" protobuf:"bytes,1,opt,name=podGroupTemplateRef"`

	// SchedulingPolicy defines the scheduling policy for this instance of the PodGroup.
	// Controllers are expected to fill this field by copying it from a PodGroupTemplate.
	// This field is immutable.
	//
	// +required
	// +k8s:immutable
	SchedulingPolicy PodGroupSchedulingPolicy `json:"schedulingPolicy" protobuf:"bytes,2,opt,name=schedulingPolicy"`

	// SchedulingConstraints defines optional scheduling constraints (e.g. topology) for this PodGroup.
	// Controllers are expected to fill this field by copying it from a PodGroupTemplate.
	// This field is immutable.
	// This field is only available when the TopologyAwareWorkloadScheduling feature gate is enabled.
	//
	// +featureGate=TopologyAwareWorkloadScheduling
	// +optional
	// +k8s:ifDisabled(TopologyAwareWorkloadScheduling)=+k8s:forbidden
	// +k8s:ifEnabled(TopologyAwareWorkloadScheduling)=+k8s:optional
	// +k8s:ifEnabled(TopologyAwareWorkloadScheduling)=+k8s:immutable
	SchedulingConstraints *PodGroupSchedulingConstraints `json:"schedulingConstraints,omitempty" protobuf:"bytes,3,opt,name=schedulingConstraints"`

	// ResourceClaims defines which ResourceClaims may be shared among Pods in
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
	ResourceClaims []PodGroupResourceClaim `json:"resourceClaims,omitempty" patchStrategy:"merge,retainKeys" patchMergeKey:"name" protobuf:"bytes,4,rep,name=resourceClaims"`

	// DisruptionMode defines the mode in which a given PodGroup can be disrupted.
	// Controllers are expected to fill this field by copying it from a PodGroupTemplate.
	// One of Pod, PodGroup. Defaults to Pod if unset.
	// This field is immutable.
	// This field is available only when the WorkloadAwarePreemption feature gate
	// is enabled.
	//
	// +featureGate=WorkloadAwarePreemption
	// +optional
	// +k8s:ifDisabled("WorkloadAwarePreemption")=+k8s:forbidden
	// +k8s:ifEnabled("WorkloadAwarePreemption")=+k8s:optional
	// +k8s:ifEnabled("WorkloadAwarePreemption")=+k8s:immutable
	// +default="Pod"
	DisruptionMode *DisruptionMode `json:"disruptionMode,omitempty" protobuf:"bytes,5,opt,name=disruptionMode,casttype=DisruptionMode"`

	// PriorityClassName defines the priority that should be considered when scheduling this pod group.
	// Controllers are expected to fill this field by copying it from a PodGroupTemplate.
	// Otherwise, it is validated and resolved similarly to the PriorityClassName on PodGroupTemplate
	// (i.e. if no priority class is specified, admission control can set this to the global default
	// priority class if it exists. Otherwise, the pod group's priority will be zero).
	// This field is immutable.
	// This field is available only when the WorkloadAwarePreemption feature gate
	// is enabled.
	//
	// +featureGate=WorkloadAwarePreemption
	// +optional
	// +k8s:ifDisabled("WorkloadAwarePreemption")=+k8s:forbidden
	// +k8s:ifEnabled("WorkloadAwarePreemption")=+k8s:optional
	// +k8s:ifEnabled("WorkloadAwarePreemption")=+k8s:format=k8s-long-name
	// +k8s:ifEnabled("WorkloadAwarePreemption")=+k8s:immutable
	PriorityClassName string `json:"priorityClassName,omitempty" protobuf:"bytes,6,opt,name=priorityClassName"`

	// Priority is the value of priority of this pod group. Various system components
	// use this field to find the priority of the pod group. When Priority Admission
	// Controller is enabled, it prevents users from setting this field. The admission
	// controller populates this field from PriorityClassName.
	// The higher the value, the higher the priority.
	// This field is immutable.
	// This field is available only when the WorkloadAwarePreemption feature gate
	// is enabled.
	//
	// +featureGate=WorkloadAwarePreemption
	// +optional
	// +k8s:ifDisabled("WorkloadAwarePreemption")=+k8s:forbidden
	// +k8s:ifEnabled("WorkloadAwarePreemption")=+k8s:optional
	// +k8s:ifEnabled("WorkloadAwarePreemption")=+k8s:immutable
	// +k8s:ifEnabled("WorkloadAwarePreemption")=+k8s:maximum=1000000000 # HighestUserDefinablePriority
	Priority *int32 `json:"priority,omitempty" protobuf:"varint,7,opt,name=priority"`
}

// PodGroupStatus represents information about the status of a pod group.
type PodGroupStatus struct {
	// Conditions represent the latest observations of the PodGroup's state.
	//
	// Known condition types:
	// - "PodGroupScheduled": Indicates whether the scheduling requirement has been satisfied.
	// - "DisruptionTarget": Indicates whether the PodGroup is about to be terminated
	//   due to disruption such as preemption.
	//
	// Known reasons for the PodGroupScheduled condition:
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
	Conditions []metav1.Condition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type" protobuf:"bytes,1,rep,name=conditions"`

	// Status of resource claims.
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
	// PodGroupScheduled represents status of the scheduling process for this PodGroup.
	PodGroupScheduled string = "PodGroupScheduled"
	// DisruptionTarget indicates the PodGroup is about to be terminated due to disruption
	// such as preemption.
	DisruptionTarget string = "DisruptionTarget"
)

// Well-known condition reasons for PodGroups.
const (
	// Unschedulable reason in the PodGroupScheduled condition indicates that the PodGroup cannot be scheduled
	// due to resource constraints, affinity/anti-affinity rules, or insufficient capacity for the PodGroup.
	PodGroupReasonUnschedulable string = "Unschedulable"
	// SchedulerError reason in the PodGroupScheduled condition means that some internal error happens
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
	// Name uniquely identifies this resource claim inside the PodGroup. This
	// must match the name of an entry in podgroup.spec.resourceClaims, which
	// implies that the string must be a DNS_LABEL.
	//
	// +required
	Name string `json:"name" protobuf:"bytes,1,name=name"`

	// ResourceClaimName is the name of the ResourceClaim that was generated for
	// the PodGroup in the namespace of the PodGroup. If this is unset, then
	// generating a ResourceClaim was not necessary. The
	// podgroup.spec.resourceClaims entry can be ignored in this case.
	//
	// +optional
	// +k8s:optional
	// +k8s:format=k8s-long-name
	ResourceClaimName *string `json:"resourceClaimName,omitempty" protobuf:"bytes,2,opt,name=resourceClaimName"`
}

// PodGroupTemplateReference references a PodGroup template defined in some object (e.g. Workload).
// Exactly one reference must be set.
// +union
type PodGroupTemplateReference struct {
	// Workload references the PodGroupTemplate within the Workload object that was used to create
	// the PodGroup.
	//
	// +optional
	// +k8s:optional
	// +k8s:unionMember
	Workload *WorkloadPodGroupTemplateReference `json:"workload" protobuf:"bytes,1,opt,name=workload"`
}

// WorkloadPodGroupTemplateReference references the PodGroupTemplate within the Workload object.
type WorkloadPodGroupTemplateReference struct {
	// WorkloadName defines the name of the Workload object.
	//
	// +required
	// +k8s:required
	// +k8s:format=k8s-long-name
	WorkloadName string `json:"workloadName" protobuf:"bytes,1,opt,name=workloadName"`

	// PodGroupTemplateName defines the PodGroupTemplate name within the Workload object.
	//
	// +required
	// +k8s:required
	// +k8s:format=k8s-short-name
	PodGroupTemplateName string `json:"podGroupTemplateName" protobuf:"bytes,2,opt,name=podGroupTemplateName"`
}

// PodGroupSchedulingConstraints defines scheduling constraints (e.g. topology) for a PodGroup.
type PodGroupSchedulingConstraints struct {
	// Topology defines the topology constraints for the pod group.
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
	// Key specifies the key of the node label representing the topology domain.
	// All pods within the PodGroup must be colocated within the same domain instance.
	// Different PodGroups can land on different domain instances even if they derive from the same PodGroupTemplate.
	// Examples: "topology.kubernetes.io/rack"
	//
	// +required
	// +k8s:required
	// +k8s:format=k8s-label-key
	Key string `json:"key" protobuf:"bytes,1,opt,name=key"`
}
