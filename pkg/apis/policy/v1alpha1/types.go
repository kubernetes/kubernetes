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
	"k8s.io/kubernetes/pkg/api/unversioned"

	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/util/intstr"
)

// PodDisruptionBudgetSpec is a description of a PodDisruptionBudget.
type PodDisruptionBudgetSpec struct {
	// The minimum number of pods that must be available simultaneously.  This
	// can be either an integer or a string specifying a percentage, e.g. "28%".
	MinAvailable intstr.IntOrString `json:"minAvailable,omitempty" protobuf:"bytes,1,opt,name=minAvailable"`

	// Label query over pods whose evictions are managed by the disruption
	// budget.
	Selector *unversioned.LabelSelector `json:"selector,omitempty" protobuf:"bytes,2,opt,name=selector"`
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
// +noMethods=true

// PodDisruptionBudget is an object to define the max disruption that can be caused to a collection of pods
type PodDisruptionBudget struct {
	unversioned.TypeMeta `json:",inline"`
	v1.ObjectMeta        `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Specification of the desired behavior of the PodDisruptionBudget.
	Spec PodDisruptionBudgetSpec `json:"spec,omitempty" protobuf:"bytes,2,opt,name=spec"`
	// Most recently observed status of the PodDisruptionBudget.
	Status PodDisruptionBudgetStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

// PodDisruptionBudgetList is a collection of PodDisruptionBudgets.
type PodDisruptionBudgetList struct {
	unversioned.TypeMeta `json:",inline"`
	unversioned.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`
	Items                []PodDisruptionBudget `json:"items" protobuf:"bytes,2,rep,name=items"`
}
