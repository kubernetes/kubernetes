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

package v1beta1

import (
	corev1 "k8s.io/api/core/v1"
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

// NodeLabelArgs holds arguments used to configure the NodeLabel plugin.
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
	// IgnoredResourceGroups defines the list of resource groups that NodeResources fit filter should ignore.
	// e.g. if group is ["example.com"], it will ignore all resource names that begin
	// with "example.com", such as "example.com/aaa" and "example.com/bbb".
	// A resource group name can't contain '/'.
	// +listType=atomic
	IgnoredResourceGroups []string `json:"ignoredResourceGroups,omitempty"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// PodTopologySpreadArgs holds arguments used to configure the PodTopologySpread plugin.
type PodTopologySpreadArgs struct {
	metav1.TypeMeta `json:",inline"`

	// DefaultConstraints defines topology spread constraints to be applied to
	// pods that don't define any in `pod.spec.topologySpreadConstraints`.
	// `topologySpreadConstraint.labelSelectors` must be empty, as they are
	// deduced the pods' membership to Services, Replication Controllers, Replica
	// Sets or Stateful Sets.
	// Empty by default.
	// +optional
	// +listType=atomic
	DefaultConstraints []corev1.TopologySpreadConstraint `json:"defaultConstraints"`
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

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// NodeResourcesLeastAllocatedArgs holds arguments used to configure NodeResourcesLeastAllocated plugin.
type NodeResourcesLeastAllocatedArgs struct {
	metav1.TypeMeta `json:",inline"`

	// Resources to be managed, if no resource is provided, default resource set with both
	// the weight of "cpu" and "memory" set to "1" will be applied.
	// Resource with "0" weight will not accountable for the final score.
	// +listType=atomic
	Resources []ResourceSpec `json:"resources,omitempty"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// NodeResourcesMostAllocatedArgs holds arguments used to configure NodeResourcesMostAllocated plugin.
type NodeResourcesMostAllocatedArgs struct {
	metav1.TypeMeta `json:",inline"`

	// Resources to be managed, if no resource is provided, default resource set with both
	// the weight of "cpu" and "memory" set to "1" will be applied.
	// Resource with "0" weight will not accountable for the final score.
	// +listType=atomic
	Resources []ResourceSpec `json:"resources,omitempty"`
}

// UtilizationShapePoint represents single point of priority function shape.
type UtilizationShapePoint struct {
	// Utilization (x axis). Valid values are 0 to 100. Fully utilized node maps to 100.
	Utilization int32 `json:"utilization"`
	// Score assigned to given utilization (y axis). Valid values are 0 to 10.
	Score int32 `json:"score"`
}

// ResourceSpec represents single resource and weight for bin packing of priority RequestedToCapacityRatioArguments.
type ResourceSpec struct {
	// Name of the resource to be managed by RequestedToCapacityRatio function.
	Name string `json:"name"`
	// Weight of the resource.
	Weight int64 `json:"weight,omitempty"`
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

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// VolumeBindingArgs holds arguments used to configure the VolumeBinding plugin.
type VolumeBindingArgs struct {
	metav1.TypeMeta `json:",inline"`

	// BindTimeoutSeconds is the timeout in seconds in volume binding operation.
	// Value must be non-negative integer. The value zero indicates no waiting.
	// If this value is nil, the default value (600) will be used.
	BindTimeoutSeconds *int64 `json:"bindTimeoutSeconds,omitempty"`
}
