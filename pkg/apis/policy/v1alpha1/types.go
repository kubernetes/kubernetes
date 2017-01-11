/*
Copyright 2016 The Kubernetes Authors.

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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/util/intstr"
)

// PodDisruptionBudgetSpec is a description of a PodDisruptionBudget.
type PodDisruptionBudgetSpec struct {
	// An eviction is allowed if at least "minAvailable" pods selected by
	// "selector" will still be available after the eviction, i.e. even in the
	// absence of the evicted pod.  So for example you can prevent all voluntary
	// evictions by specifying "100%".
	// +optional
	MinAvailable intstr.IntOrString `json:"minAvailable,omitempty" protobuf:"bytes,1,opt,name=minAvailable"`

	// Label query over pods whose evictions are managed by the disruption
	// budget.
	// +optional
	Selector *metav1.LabelSelector `json:"selector,omitempty" protobuf:"bytes,2,opt,name=selector"`
}

// PodDisruptionBudgetStatus represents information about the status of a
// PodDisruptionBudget. Status may trail the actual state of a system.
type PodDisruptionBudgetStatus struct {
	// Whether or not a disruption is currently allowed.
	PodDisruptionAllowed bool `json:"disruptionAllowed" protobuf:"varint,1,opt,name=disruptionAllowed"`

	// current number of healthy pods
	CurrentHealthy int32 `json:"currentHealthy" protobuf:"varint,2,opt,name=currentHealthy"`

	// minimum desired number of healthy pods
	DesiredHealthy int32 `json:"desiredHealthy" protobuf:"varint,3,opt,name=desiredHealthy"`

	// total number of pods counted by this disruption budget
	ExpectedPods int32 `json:"expectedPods" protobuf:"varint,4,opt,name=expectedPods"`
}

// +genclient=true

// PodDisruptionBudget is an object to define the max disruption that can be caused to a collection of pods
type PodDisruptionBudget struct {
	metav1.TypeMeta `json:",inline"`
	// +optional
	v1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Specification of the desired behavior of the PodDisruptionBudget.
	// +optional
	Spec PodDisruptionBudgetSpec `json:"spec,omitempty" protobuf:"bytes,2,opt,name=spec"`
	// Most recently observed status of the PodDisruptionBudget.
	// +optional
	Status PodDisruptionBudgetStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

// PodDisruptionBudgetList is a collection of PodDisruptionBudgets.
type PodDisruptionBudgetList struct {
	metav1.TypeMeta `json:",inline"`
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`
	Items           []PodDisruptionBudget `json:"items" protobuf:"bytes,2,rep,name=items"`
}

// Eviction evicts a pod from its node subject to certain policies and safety constraints.
// This is a subresource of Pod.  A request to cause such an eviction is
// created by POSTing to .../pods/<pod name>/eviction.
type Eviction struct {
	metav1.TypeMeta `json:",inline"`

	// ObjectMeta describes the pod that is being evicted.
	// +optional
	v1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// DeleteOptions may be provided
	// +optional
	DeleteOptions *v1.DeleteOptions `json:"deleteOptions,omitempty" protobuf:"bytes,2,opt,name=deleteOptions"`
}
