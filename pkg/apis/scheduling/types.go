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

	// PodGroupProtectionFinalizer is the finalizer added to PodGroups to prevent
	// premature deletion while pods still reference them.
	PodGroupProtectionFinalizer = GroupName + "/podgroup-protection"
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
// when managing the lifecycle of workloads from the scheduling perspective,
// including scheduling, preemption, eviction and other phases.
// Workload API enablement is toggled by the GenericWorkload feature gate.
type Workload struct {
	metav1.TypeMeta
	// Standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
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
	ControllerRef *TypedLocalObjectReference

	// PodGroupTemplates is the list of templates that make up the Workload.
	// The maximum number of templates is 8. This field is immutable.
	//
	// +required
	// +listType=map
	// +listMapKey=name
	PodGroupTemplates []PodGroupTemplate
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

// PodGroupTemplate represents a template for a set of pods with a scheduling policy.
type PodGroupTemplate struct {
	// Name is a unique identifier for the PodGroupTemplate within the Workload.
	// It must be a DNS label. This field is immutable.
	//
	// +required
	Name string

	// SchedulingPolicy defines the scheduling policy for this PodGroupTemplate.
	//
	// +required
	SchedulingPolicy PodGroupSchedulingPolicy

	// SchedulingConstraints defines optional scheduling constraints (e.g. topology) for this PodGroupTemplate.
	// This field is only available when the TopologyAwareWorkloadScheduling feature gate is enabled.
	//
	// +optional
	// +featureGate=TopologyAwareWorkloadScheduling
	SchedulingConstraints *PodGroupSchedulingConstraints
}

// PodGroupSchedulingPolicy defines the scheduling configuration for a PodGroup.
// Exactly one policy must be set.
// +union
type PodGroupSchedulingPolicy struct {
	// Basic specifies that the pods in this group should be scheduled using
	// standard Kubernetes scheduling behavior.
	//
	// +optional
	Basic *BasicSchedulingPolicy

	// Gang specifies that the pods in this group should be scheduled using
	// all-or-nothing semantics.
	//
	// +optional
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

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// PodGroup represents a runtime instance of pods grouped together.
// PodGroups are created by workload controllers (Job, LWS, JobSet, etc...) from
// Workload.podGroupTemplates.
// PodGroup API enablement is toggled by the GenericWorkload feature gate.
type PodGroup struct {
	metav1.TypeMeta
	// Standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	//
	// +optional
	metav1.ObjectMeta

	// Spec defines the desired state of the PodGroup.
	//
	// +required
	Spec PodGroupSpec

	// Status represents the current observed state of the PodGroup.
	//
	// +optional
	Status PodGroupStatus
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// PodGroupList contains a list of PodGroup resources.
type PodGroupList struct {
	metav1.TypeMeta
	// Standard list metadata.
	//
	// +optional
	metav1.ListMeta

	// Items is the list of PodGroups.
	Items []PodGroup
}

// PodGroupSpec defines the desired state of a PodGroup.
type PodGroupSpec struct {
	// PodGroupTemplateRef references an optional PodGroup template within other object
	// (e.g. Workload) that was used to create the PodGroup. This field is immutable.
	//
	// +optional
	PodGroupTemplateRef *PodGroupTemplateReference

	// SchedulingPolicy defines the scheduling policy for this instance of the PodGroup.
	// Controllers are expected to fill this field by copying it from a PodGroupTemplate.
	// This field is immutable.
	//
	// +required
	SchedulingPolicy PodGroupSchedulingPolicy

	// SchedulingConstraints defines optional scheduling constraints (e.g. topology) for this PodGroup.
	// Controllers are expected to fill this field by copying it from a PodGroupTemplate.
	// This field is immutable.
	// This field is only available when the TopologyAwareWorkloadScheduling feature gate is enabled.
	//
	// +optional
	// +featureGate=TopologyAwareWorkloadScheduling
	SchedulingConstraints *PodGroupSchedulingConstraints
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
	Conditions []metav1.Condition
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

// PodGroupTemplateReference references a PodGroup template defined in some object (e.g. Workload).
// Exactly one reference must be set.
// +union
type PodGroupTemplateReference struct {
	// Workload references the PodGroupTemplate within the Workload object that was used to create
	// the PodGroup.
	//
	// +optional
	Workload *WorkloadPodGroupTemplateReference
}

// WorkloadPodGroupTemplateReference references the PodGroupTemplate within the Workload object.
type WorkloadPodGroupTemplateReference struct {
	// WorkloadName defines the name of the Workload object.
	//
	// +required
	WorkloadName string

	// PodGroupTemplateName defines the PodGroupTemplate name within the Workload object.
	//
	// +required
	PodGroupTemplateName string
}

// PodGroupSchedulingConstraints defines scheduling constraints (e.g. topology) for a PodGroup.
type PodGroupSchedulingConstraints struct {
	// Topology defines the topology constraints for the pod group.
	// Currently only a single topology constraint can be specified. This may change in the future.
	//
	// +optional
	// +listType=atomic
	Topology []TopologyConstraint
}

// TopologyConstraint defines a topology constraint for a PodGroup.
type TopologyConstraint struct {
	// Key specifies the key of the node label representing the topology domain.
	// All pods within the PodGroup must be colocated within the same domain instance.
	// Different PodGroups can land on different domain instances even if they derive from the same PodGroupTemplate.
	// Examples: "topology.kubernetes.io/rack"
	//
	// +required
	Key string
}
