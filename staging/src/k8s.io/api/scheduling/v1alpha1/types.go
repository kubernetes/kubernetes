/*
Copyright 2017 The Kubernetes Authors.

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

package v1alpha1

import (
	apiv1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// DEPRECATED - This group version of PriorityClass is deprecated by scheduling.k8s.io/v1/PriorityClass.
// PriorityClass defines mapping from a priority class name to the priority
// integer value. The value can be any valid integer.
type PriorityClass struct {
	metav1.TypeMeta `json:",inline"`
	// Standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// value represents the integer value of this priority class. This is the actual priority that pods
	// receive when they have the name of this class in their pod spec.
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

// PriorityClassList is a collection of priority classes.
type PriorityClassList struct {
	metav1.TypeMeta `json:",inline"`
	// Standard list metadata
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// items is the list of PriorityClasses
	Items []PriorityClass `json:"items" protobuf:"bytes,2,rep,name=items"`
}

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

// WorkloadMaxPodGroups is the maximum number of pod groups per Workload.
const WorkloadMaxPodGroups = 8

// WorkloadSpec defines the desired state of a Workload.
type WorkloadSpec struct {
	// ControllerRef is an optional reference to the controlling object, such as a
	// Deployment or Job. This field is intended for use by tools like CLIs
	// to provide a link back to the original workload definition.
	// When set, it cannot be changed.
	//
	// +optional
	ControllerRef *TypedLocalObjectReference `json:"controllerRef,omitempty" protobuf:"bytes,1,opt,name=controllerRef"`

	// PodGroups is the list of pod groups that make up the Workload.
	// The maximum number of pod groups is 8. This field is immutable.
	//
	// +required
	// +listType=map
	// +listMapKey=name
	PodGroups []PodGroup `json:"podGroups" protobuf:"bytes,2,rep,name=podGroups"`
}

// TypedLocalObjectReference allows to reference typed object inside the same namespace.
type TypedLocalObjectReference struct {
	// APIGroup is the group for the resource being referenced.
	// If APIGroup is empty, the specified Kind must be in the core API group.
	// For any other third-party types, setting APIGroup is required.
	// It must be a DNS subdomain.
	//
	// +optional
	APIGroup string `json:"apiGroup,omitempty" protobuf:"bytes,1,opt,name=apiGroup"`
	// Kind is the type of resource being referenced.
	// It must be a path segment name.
	//
	// +required
	Kind string `json:"kind" protobuf:"bytes,2,opt,name=kind"`
	// Name is the name of resource being referenced.
	// It must be a path segment name.
	//
	// +required
	Name string `json:"name" protobuf:"bytes,3,opt,name=name"`
}

// PodGroup represents a set of pods with a common scheduling policy.
type PodGroup struct {
	// Name is a unique identifier for the PodGroup within the Workload.
	// It must be a DNS label. This field is immutable.
	//
	// +required
	Name string `json:"name" protobuf:"bytes,1,opt,name=name"`

	// Policy defines the scheduling policy for this PodGroup.
	//
	// +required
	Policy PodGroupPolicy `json:"policy" protobuf:"bytes,3,opt,name=policy"`
}

// PodGroupPolicy defines the scheduling configuration for a PodGroup.
type PodGroupPolicy struct {
	// Basic specifies that the pods in this group should be scheduled using
	// standard Kubernetes scheduling behavior.
	//
	// +optional
	// +oneOf=PolicySelection
	Basic *BasicSchedulingPolicy `json:"basic,omitempty" protobuf:"bytes,2,opt,name=basic"`

	// Gang specifies that the pods in this group should be scheduled using
	// all-or-nothing semantics.
	//
	// +optional
	// +oneOf=PolicySelection
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
	MinCount int32 `json:"minCount" protobuf:"varint,1,opt,name=minCount"`
}
