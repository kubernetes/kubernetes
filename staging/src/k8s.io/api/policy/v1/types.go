/*
Copyright 2021 The Kubernetes Authors.

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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
)

// PodDisruptionBudgetSpec is a description of a PodDisruptionBudget.
type PodDisruptionBudgetSpec struct {
	// An eviction is allowed if at least "minAvailable" pods selected by
	// "selector" will still be available after the eviction, i.e. even in the
	// absence of the evicted pod.  So for example you can prevent all voluntary
	// evictions by specifying "100%".
	// +optional
	MinAvailable *intstr.IntOrString `json:"minAvailable,omitempty" protobuf:"bytes,1,opt,name=minAvailable"`

	// Label query over pods whose evictions are managed by the disruption
	// budget.
	// A null selector will match no pods, while an empty ({}) selector will select
	// all pods within the namespace.
	// +patchStrategy=replace
	// +optional
	Selector *metav1.LabelSelector `json:"selector,omitempty" patchStrategy:"replace" protobuf:"bytes,2,opt,name=selector"`

	// An eviction is allowed if at most "maxUnavailable" pods selected by
	// "selector" are unavailable after the eviction, i.e. even in absence of
	// the evicted pod. For example, one can prevent all voluntary evictions
	// by specifying 0. This is a mutually exclusive setting with "minAvailable".
	// +optional
	MaxUnavailable *intstr.IntOrString `json:"maxUnavailable,omitempty" protobuf:"bytes,3,opt,name=maxUnavailable"`
}

// PodDisruptionBudgetStatus represents information about the status of a
// PodDisruptionBudget. Status may trail the actual state of a system.
type PodDisruptionBudgetStatus struct {
	// Most recent generation observed when updating this PDB status. DisruptionsAllowed and other
	// status information is valid only if observedGeneration equals to PDB's object generation.
	// +optional
	ObservedGeneration int64 `json:"observedGeneration,omitempty" protobuf:"varint,1,opt,name=observedGeneration"`

	// DisruptedPods contains information about pods whose eviction was
	// processed by the API server eviction subresource handler but has not
	// yet been observed by the PodDisruptionBudget controller.
	// A pod will be in this map from the time when the API server processed the
	// eviction request to the time when the pod is seen by PDB controller
	// as having been marked for deletion (or after a timeout). The key in the map is the name of the pod
	// and the value is the time when the API server processed the eviction request. If
	// the deletion didn't occur and a pod is still there it will be removed from
	// the list automatically by PodDisruptionBudget controller after some time.
	// If everything goes smooth this map should be empty for the most of the time.
	// Large number of entries in the map may indicate problems with pod deletions.
	// +optional
	DisruptedPods map[string]metav1.Time `json:"disruptedPods,omitempty" protobuf:"bytes,2,rep,name=disruptedPods"`

	// Number of pod disruptions that are currently allowed.
	DisruptionsAllowed int32 `json:"disruptionsAllowed" protobuf:"varint,3,opt,name=disruptionsAllowed"`

	// current number of healthy pods
	CurrentHealthy int32 `json:"currentHealthy" protobuf:"varint,4,opt,name=currentHealthy"`

	// minimum desired number of healthy pods
	DesiredHealthy int32 `json:"desiredHealthy" protobuf:"varint,5,opt,name=desiredHealthy"`

	// total number of pods counted by this disruption budget
	ExpectedPods int32 `json:"expectedPods" protobuf:"varint,6,opt,name=expectedPods"`

	// Conditions contain conditions for PDB. The disruption controller sets the
	// DisruptionAllowed condition. The following are known values for the reason field
	// (additional reasons could be added in the future):
	// - SyncFailed: The controller encountered an error and wasn't able to compute
	//               the number of allowed disruptions. Therefore no disruptions are
	//               allowed and the status of the condition will be False.
	// - InsufficientPods: The number of pods are either at or below the number
	//                     required by the PodDisruptionBudget. No disruptions are
	//                     allowed and the status of the condition will be False.
	// - SufficientPods: There are more pods than required by the PodDisruptionBudget.
	//                   The condition will be True, and the number of allowed
	//                   disruptions are provided by the disruptionsAllowed property.
	//
	// +optional
	// +patchMergeKey=type
	// +patchStrategy=merge
	// +listType=map
	// +listMapKey=type
	Conditions []metav1.Condition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type" protobuf:"bytes,7,rep,name=conditions"`
}

const (
	// DisruptionAllowedCondition is a condition set by the disruption controller
	// that signal whether any of the pods covered by the PDB can be disrupted.
	DisruptionAllowedCondition = "DisruptionAllowed"

	// SyncFailedReason is set on the DisruptionAllowed condition if reconcile
	// of the PDB failed and therefore disruption of pods are not allowed.
	SyncFailedReason = "SyncFailed"
	// SufficientPodsReason is set on the DisruptionAllowed condition if there are
	// more pods covered by the PDB than required and at least one can be disrupted.
	SufficientPodsReason = "SufficientPods"
	// InsufficientPodsReason is set on the DisruptionAllowed condition if the number
	// of pods are equal to or fewer than required by the PDB.
	InsufficientPodsReason = "InsufficientPods"
)

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// PodDisruptionBudget is an object to define the max disruption that can be caused to a collection of pods
type PodDisruptionBudget struct {
	metav1.TypeMeta `json:",inline"`
	// Standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Specification of the desired behavior of the PodDisruptionBudget.
	// +optional
	Spec PodDisruptionBudgetSpec `json:"spec,omitempty" protobuf:"bytes,2,opt,name=spec"`
	// Most recently observed status of the PodDisruptionBudget.
	// +optional
	Status PodDisruptionBudgetStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// PodDisruptionBudgetList is a collection of PodDisruptionBudgets.
type PodDisruptionBudgetList struct {
	metav1.TypeMeta `json:",inline"`
	// Standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`
	// Items is a list of PodDisruptionBudgets
	Items []PodDisruptionBudget `json:"items" protobuf:"bytes,2,rep,name=items"`
}
