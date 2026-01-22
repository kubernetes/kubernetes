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

package scheduling

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/apis/core"
)

const (
	// DefaultPriorityWhenNoDefaultClassExists is used to set priority of pods
	// that do not specify any priority class and there is no priority class
	// marked as default.
	DefaultPriorityWhenNoDefaultClassExists = 0
	// HighestUserDefinablePriority is the highest priority for user defined priority classes. Priority values larger than 1 billion are reserved for Kubernetes system use.
	HighestUserDefinablePriority = int32(1000000000)
	// SystemCriticalPriority is the beginning of the range of priority values for critical system components.
	SystemCriticalPriority = 2 * HighestUserDefinablePriority
	// SystemPriorityClassPrefix is the prefix reserved for system priority class names. Other priority
	// classes are not allowed to start with this prefix.
	// NOTE: In order to avoid conflict of names with user-defined priority classes, all the names must
	// start with SystemPriorityClassPrefix.
	SystemPriorityClassPrefix = "system-"
	// SystemClusterCritical is the system priority class name that represents cluster-critical.
	SystemClusterCritical = SystemPriorityClassPrefix + "cluster-critical"
	// SystemNodeCritical is the system priority class name that represents node-critical.
	SystemNodeCritical = SystemPriorityClassPrefix + "node-critical"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// PriorityClass defines the mapping from a priority class name to the priority
// integer value. The value can be any valid integer.
type PriorityClass struct {
	metav1.TypeMeta
	// Standard object metadata; More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata.
	// +optional
	metav1.ObjectMeta

	// value represents the integer value of this priority class. This is the actual priority that pods
	// receive when they have the name of this class in their pod spec.
	Value int32

	// globalDefault specifies whether this PriorityClass should be considered as
	// the default priority for pods that do not have any priority class.
	// Only one PriorityClass can be marked as `globalDefault`. However, if more than
	// one PriorityClasses exists with their `globalDefault` field set to true,
	// the smallest value of such global default PriorityClasses will be used as the default priority.
	// +optional
	GlobalDefault bool

	// description is an arbitrary string that usually provides guidelines on
	// when this priority class should be used.
	// +optional
	Description string

	// preemptionPolicy it the Policy for preempting pods with lower priority.
	// One of Never, PreemptLowerPriority.
	// Defaults to PreemptLowerPriority if unset.
	// +optional
	PreemptionPolicy *core.PreemptionPolicy
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// PriorityClassList is a collection of priority classes.
type PriorityClassList struct {
	metav1.TypeMeta
	// Standard list metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#types-kinds
	// +optional
	metav1.ListMeta

	// items is the list of PriorityClasses.
	Items []PriorityClass
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Workload allows for expressing scheduling constraints that should be used
// when managing lifecycle of workloads from scheduling perspective,
// including scheduling, preemption, eviction and other phases.
type Workload struct {
	metav1.TypeMeta
	// Standard object's metadata.
	// Name must be a DNS subdomain.
	//
	// +optional
	metav1.ObjectMeta

	// Spec defines the desired behavior of a Workload.
	//
	// +required
	Spec WorkloadSpec
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// WorkloadList contains a list of Workload resources.
type WorkloadList struct {
	metav1.TypeMeta
	// Standard list metadata.
	//
	// +optional
	metav1.ListMeta

	// Items is the list of Workloads.
	Items []Workload
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
	ControllerRef *TypedLocalObjectReference

	// PodGroups is the list of pod groups that make up the Workload.
	// The maximum number of pod groups is 8. This field is immutable.
	//
	// +required
	// +listType=map
	// +listMapKey=name
	PodGroups []PodGroup
}

// TypedLocalObjectReference allows to reference typed object inside the same namespace.
type TypedLocalObjectReference struct {
	// APIGroup is the group for the resource being referenced.
	// If APIGroup is empty, the specified Kind must be in the core API group.
	// For any other third-party types, setting APIGroup is required.
	// It must be a DNS subdomain.
	//
	// +optional
	APIGroup string
	// Kind is the type of resource being referenced.
	// It must be a path segment name.
	//
	// +required
	Kind string
	// Name is the name of resource being referenced.
	// It must be a path segment name.
	//
	// +required
	Name string
}

// PodGroup represents a set of pods with a common scheduling policy.
type PodGroup struct {
	// Name is a unique identifier for the PodGroup within the Workload.
	// It must be a DNS label. This field is immutable.
	//
	// +required
	Name string

	// Policy defines the scheduling policy for this PodGroup.
	//
	// +required
	Policy PodGroupPolicy
}

// PodGroupPolicy defines the scheduling configuration for a PodGroup.
type PodGroupPolicy struct {
	// Basic specifies that the pods in this group should be scheduled using
	// standard Kubernetes scheduling behavior.
	//
	// +optional
	// +oneOf=PolicySelection
	Basic *BasicSchedulingPolicy

	// Gang specifies that the pods in this group should be scheduled using
	// all-or-nothing semantics.
	//
	// +optional
	// +oneOf=PolicySelection
	Gang *GangSchedulingPolicy
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
	MinCount int32
}
