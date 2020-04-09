/*
Copyright 2020 The Kubernetes Authors.

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

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// InterPodAffinityArgs holds arguments used to configure the InterPodAffinity plugin.
type InterPodAffinityArgs struct {
	metav1.TypeMeta `json:",inline"`

	// HardPodAffinityWeight is the scoring weight for existing pods with a
	// matching hard affinity to the incoming pod.
	HardPodAffinityWeight *int32 `json:"hardPodAffinityWeight,omitempty"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// NodeLabelArgs holds arguments that used to configure the NodeLabel plugin.
type NodeLabelArgs struct {
	metav1.TypeMeta `json:",inline"`

	// PresentLabels should be present for the node to be considered a fit for hosting the pod
	// +listType=atomic
	PresentLabels []string `json:"presentLabels,omitempty"`
	// AbsentLabels should be absent for the node to be considered a fit for hosting the pod
	// +listType=atomic
	AbsentLabels []string `json:"absentLabels,omitempty"`
	// Nodes that have labels in the list will get a higher score.
	// +listType=atomic
	PresentLabelsPreference []string `json:"presentLabelsPreference,omitempty"`
	// Nodes that don't have labels in the list will get a higher score.
	// +listType=atomic
	AbsentLabelsPreference []string `json:"absentLabelsPreference,omitempty"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// NodeResourcesFitArgs holds arguments used to configure the NodeResourcesFit plugin.
type NodeResourcesFitArgs struct {
	metav1.TypeMeta `json:",inline"`

	// IgnoredResources is the list of resources that NodeResources fit filter
	// should ignore.
	// +listType=atomic
	IgnoredResources []string `json:"ignoredResources,omitempty"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// RequestedToCapacityRatioArgs holds arguments used to configure RequestedToCapacityRatio plugin.
type RequestedToCapacityRatioArgs struct {
	metav1.TypeMeta `json:",inline"`

	// Points defining priority function shape
	// +listType=atomic
	Shape []UtilizationShapePoint `json:"shape"`
	// Resources to be managed
	// +listType=atomic
	Resources []ResourceSpec `json:"resources,omitempty"`
}

// TODO add JSON tags and backward compatible conversion in v1beta1.
// UtilizationShapePoint and ResourceSpec fields are not annotated with JSON tags in v1alpha2
// to maintain backward compatibility with the args shipped with v1.18.
// See https://github.com/kubernetes/kubernetes/pull/88585#discussion_r405021905

// UtilizationShapePoint represents single point of priority function shape.
type UtilizationShapePoint struct {
	// Utilization (x axis). Valid values are 0 to 100. Fully utilized node maps to 100.
	Utilization int32
	// Score assigned to given utilization (y axis). Valid values are 0 to 10.
	Score int32
}

// ResourceSpec represents single resource and weight for bin packing of priority RequestedToCapacityRatioArguments.
type ResourceSpec struct {
	// Name of the resource to be managed by RequestedToCapacityRatio function.
	Name string
	// Weight of the resource.
	Weight int64
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ServiceAffinityArgs holds arguments used to configure the ServiceAffinity plugin.
type ServiceAffinityArgs struct {
	metav1.TypeMeta `json:",inline"`

	// AffinityLabels are homogeneous for pods that are scheduled to a node.
	// (i.e. it returns true IFF this pod can be added to this node such that all other pods in
	// the same service are running on nodes with the exact same values for Labels).
	// +listType=atomic
	AffinityLabels []string `json:"affinityLabels,omitempty"`
	// AntiAffinityLabelsPreference are the labels to consider for service anti affinity scoring.
	// +listType=atomic
	AntiAffinityLabelsPreference []string `json:"antiAffinityLabelsPreference,omitempty"`
}
