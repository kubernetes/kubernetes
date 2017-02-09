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

package policy

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
	MinAvailable intstr.IntOrString

	// Label query over pods whose evictions are managed by the disruption
	// budget.
	// +optional
	Selector *metav1.LabelSelector
}

// PodDisruptionBudgetStatus represents information about the status of a
// PodDisruptionBudget. Status may trail the actual state of a system.
type PodDisruptionBudgetStatus struct {
	// Most recent generation observed when updating this PDB status. PodDisruptionsAllowed and other
	// status informatio is valid only if observedGeneration equals to PDB's object generation.
	// +optional
	ObservedGeneration int64

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
	DisruptedPods map[string]metav1.Time

	// Number of pod disruptions that are currently allowed.
	PodDisruptionsAllowed int32

	// current number of healthy pods
	CurrentHealthy int32

	// minimum desired number of healthy pods
	DesiredHealthy int32

	// total number of pods counted by this disruption budget
	ExpectedPods int32
}

// +genclient=true

// PodDisruptionBudget is an object to define the max disruption that can be caused to a collection of pods
type PodDisruptionBudget struct {
	metav1.TypeMeta
	// +optional
	metav1.ObjectMeta

	// Specification of the desired behavior of the PodDisruptionBudget.
	// +optional
	Spec PodDisruptionBudgetSpec
	// Most recently observed status of the PodDisruptionBudget.
	// +optional
	Status PodDisruptionBudgetStatus
}

// PodDisruptionBudgetList is a collection of PodDisruptionBudgets.
type PodDisruptionBudgetList struct {
	metav1.TypeMeta
	// +optional
	metav1.ListMeta
	Items []PodDisruptionBudget
}

// +genclient=true
// +noMethods=true

// Eviction evicts a pod from its node subject to certain policies and safety constraints.
// This is a subresource of Pod.  A request to cause such an eviction is
// created by POSTing to .../pods/<pod name>/eviction.
type Eviction struct {
	metav1.TypeMeta

	// ObjectMeta describes the pod that is being evicted.
	// +optional
	metav1.ObjectMeta

	// DeleteOptions may be provided
	// +optional
	DeleteOptions *metav1.DeleteOptions
}
