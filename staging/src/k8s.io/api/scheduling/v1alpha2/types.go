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
// when managing lifecycle of workloads from scheduling perspective,
// including scheduling, preemption, eviction and other phases.
type Workload struct {
	metav1.TypeMeta `json:",inline"`
	// Standard object's metadata.
	// Name must be a DNS subdomain.
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
	metav1.TypeMeta `json:",inline"`
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
	// When set, it cannot be changed.
	//
	// +optional
	// +k8s:alpha(since:"1.36")=+k8s:optional
	// +k8s:alpha(since:"1.36")=+k8s:update=NoModify
	ControllerRef *TypedLocalObjectReference `json:"controllerRef,omitempty" protobuf:"bytes,1,opt,name=controllerRef"`

	// PodGroupTemplates is the list of templates that make up the Workload.
	// The maximum number of templates is 8. This field is immutable.
	//
	// +required
	// +listType=map
	// +listMapKey=name
	// +k8s:alpha(since:"1.36")=+k8s:required
	// +k8s:alpha(since:"1.36")=+k8s:listType=map
	// +k8s:alpha(since:"1.36")=+k8s:listMapKey=name
	// +k8s:alpha(since:"1.36")=+k8s:maxItems=8
	// +k8s:alpha(since:"1.36")=+k8s:immutable
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
	// +k8s:alpha(since:"1.36")=+k8s:optional
	// +k8s:alpha(since:"1.36")=+k8s:format=k8s-long-name
	APIGroup string `json:"apiGroup,omitempty" protobuf:"bytes,1,opt,name=apiGroup"`
	// Kind is the type of resource being referenced.
	// It must be a path segment name.
	//
	// +required
	// +k8s:alpha(since:"1.36")=+k8s:required
	// +k8s:alpha(since:"1.36")=+k8s:format=k8s-path-segment-name
	Kind string `json:"kind" protobuf:"bytes,2,opt,name=kind"`
	// Name is the name of resource being referenced.
	// It must be a path segment name.
	//
	// +required
	// +k8s:alpha(since:"1.36")=+k8s:required
	// +k8s:alpha(since:"1.36")=+k8s:format=k8s-path-segment-name
	Name string `json:"name" protobuf:"bytes,3,opt,name=name"`
}

// PodGroupTemplate represents a template for a set of pods with a common policy.
type PodGroupTemplate struct {
	// Name is a unique identifier for the PodGroup within the Workload.
	// It must be a DNS label. This field is immutable.
	//
	// +required
	// +k8s:alpha(since:"1.36")=+k8s:required
	// +k8s:alpha(since:"1.36")=+k8s:format=k8s-short-name
	Name string `json:"name" protobuf:"bytes,1,opt,name=name"`

	// SchedulingPolicy defines the scheduling policy for this PodGroupTemplate.
	//
	// +required
	SchedulingPolicy PodGroupSchedulingPolicy `json:"schedulingPolicy" protobuf:"bytes,2,opt,name=schedulingPolicy"`
}

// PodGroupSchedulingPolicy defines the scheduling configuration for a PodGroup.
// +union
type PodGroupSchedulingPolicy struct {
	// Basic specifies that the pods in this group should be scheduled using
	// standard Kubernetes scheduling behavior.
	//
	// +optional
	// +k8s:alpha(since:"1.36")=+k8s:optional
	// +oneOf=PolicySelection
	// +k8s:alpha(since:"1.36")=+k8s:unionMember
	Basic *BasicSchedulingPolicy `json:"basic,omitempty" protobuf:"bytes,2,opt,name=basic"`

	// Gang specifies that the pods in this group should be scheduled using
	// all-or-nothing semantics.
	//
	// +optional
	// +k8s:alpha(since:"1.36")=+k8s:optional
	// +oneOf=PolicySelection
	// +k8s:alpha(since:"1.36")=+k8s:unionMember
	Gang *GangSchedulingPolicy `json:"gang,omitempty" protobuf:"bytes,3,opt,name=gang"`
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
	// +k8s:alpha(since:"1.36")=+k8s:required
	// +k8s:alpha(since:"1.36")=+k8s:minimum=0
	MinCount int32 `json:"minCount" protobuf:"varint,1,opt,name=minCount"`
}

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// PodGroup represents a runtime instance of pods grouped for gang scheduling.
// PodGroups are created by workload controllers (Job, LWS, JobSet, etc...) from
// Workload.podGroupTemplates. Each PodGroup corresponds to one replica of the Workload.
type PodGroup struct {
	metav1.TypeMeta `json:",inline"`
	// Standard object's metadata.
	// Name must be a DNS subdomain.
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
	metav1.TypeMeta `json:",inline"`
	// Standard list metadata.
	//
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Items is the list of PodGroups.
	Items []PodGroup `json:"items" protobuf:"bytes,2,rep,name=items"`
}

// PodGroupSpec defines the desired state of a PodGroup.
type PodGroupSpec struct {
	// PodGroupTemplateRef references the PodGroupTemplate within the Workload object that was used to create
	// the PodGroup.
	//
	// +optional
	PodGroupTemplateRef *PodGroupTemplateReference `json:"podGroupTemplateRef,omitempty" protobuf:"bytes,1,opt,name=podGroupTemplateRef"`

	// SchedulingPolicy defines the scheduling policy for this instance of the PodGroup.
	// It is copied from the template on PodGroup creation.
	//
	// +required
	SchedulingPolicy PodGroupSchedulingPolicy `json:"schedulingPolicy" protobuf:"bytes,2,opt,name=schedulingPolicy"`
}

// PodGroupStatus represents information about the status of a pod group.
type PodGroupStatus struct {
	// Conditions represent the latest observations of the PodGroup's state.
	//
	// Known condition types:
	// - "PodGroupScheduled": Indicates whether the scheduling requirement has been satisfied.
	//   - Status=True: All required pods have been assigned to nodes.
	//   - Status=False: Scheduling failed (i.e., timeout, unschedulable, etc.).
	//
	// Known reasons for PodGroupScheduled condition:
	// - "Scheduled": All required pods have been successfully scheduled.
	// - "Unschedulable": The PodGroup cannot be scheduled due to resource constraints,
	//   affinity/anti-affinity rules, or insufficient capacity for the gang.
	// - "Preempted": The PodGroup was preempted to make room for higher-priority workloads.
	// - "Timeout": The PodGroup failed to schedule within the configured timeout.
	//
	// +optional
	// +patchMergeKey=type
	// +patchStrategy=merge
	// +listType=map
	// +listMapKey=type
	Conditions []metav1.Condition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type" protobuf:"bytes,1,rep,name=conditions"`
}

// PodGroupTemplateReference references the PodGroupTemplate within the Workload object.
type PodGroupTemplateReference struct {
	// WorkloadName defines the name of the Workload object.
	//
	// +optional
	WorkloadName string `json:"workloadName,omitempty" protobuf:"bytes,1,opt,name=workloadName"`

	// PodGroupTemplateName defines the PodGroupTemplate name within the Workload object.
	//
	// +optional
	PodGroupTemplateName string `json:"podGroupTemplateName,omitempty" protobuf:"bytes,2,opt,name=podGroupTemplateName"`
}
