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

package v1beta3

import (
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// DefaultPreemptionArgs holds arguments used to configure the
// DefaultPreemption plugin.
type DefaultPreemptionArgs struct {
	metav1.TypeMeta `json:",inline"`

	// MinCandidateNodesPercentage is the minimum number of candidates to
	// shortlist when dry running preemption as a percentage of number of nodes.
	// Must be in the range [0, 100]. Defaults to 10% of the cluster size if
	// unspecified.
	MinCandidateNodesPercentage *int32 `json:"minCandidateNodesPercentage,omitempty"`
	// MinCandidateNodesAbsolute is the absolute minimum number of candidates to
	// shortlist. The likely number of candidates enumerated for dry running
	// preemption is given by the formula:
	// numCandidates = max(numNodes * minCandidateNodesPercentage, minCandidateNodesAbsolute)
	// We say "likely" because there are other factors such as PDB violations
	// that play a role in the number of candidates shortlisted. Must be at least
	// 0 nodes. Defaults to 100 nodes if unspecified.
	MinCandidateNodesAbsolute *int32 `json:"minCandidateNodesAbsolute,omitempty"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// InterPodAffinityArgs holds arguments used to configure the InterPodAffinity plugin.
type InterPodAffinityArgs struct {
	metav1.TypeMeta `json:",inline"`

	// HardPodAffinityWeight is the scoring weight for existing pods with a
	// matching hard affinity to the incoming pod.
	HardPodAffinityWeight *int32 `json:"hardPodAffinityWeight,omitempty"`

	// IgnorePreferredTermsOfExistingPods configures the scheduler to ignore existing pods' preferred affinity
	// rules when scoring candidate nodes, unless the incoming pod has inter-pod affinities.
	IgnorePreferredTermsOfExistingPods bool `json:"ignorePreferredTermsOfExistingPods"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// NodeResourcesFitArgs holds arguments used to configure the NodeResourcesFit plugin.
type NodeResourcesFitArgs struct {
	metav1.TypeMeta `json:",inline"`

	// IgnoredResources is the list of resources that NodeResources fit filter
	// should ignore. This doesn't apply to scoring.
	// +listType=atomic
	IgnoredResources []string `json:"ignoredResources,omitempty"`
	// IgnoredResourceGroups defines the list of resource groups that NodeResources fit filter should ignore.
	// e.g. if group is ["example.com"], it will ignore all resource names that begin
	// with "example.com", such as "example.com/aaa" and "example.com/bbb".
	// A resource group name can't contain '/'. This doesn't apply to scoring.
	// +listType=atomic
	IgnoredResourceGroups []string `json:"ignoredResourceGroups,omitempty"`

	// ScoringStrategy selects the node resource scoring strategy.
	// The default strategy is LeastAllocated with an equal "cpu" and "memory" weight.
	ScoringStrategy *ScoringStrategy `json:"scoringStrategy,omitempty"`
}

// PodTopologySpreadConstraintsDefaulting defines how to set default constraints
// for the PodTopologySpread plugin.
type PodTopologySpreadConstraintsDefaulting string

const (
	// SystemDefaulting instructs to use the kubernetes defined default.
	SystemDefaulting PodTopologySpreadConstraintsDefaulting = "System"
	// ListDefaulting instructs to use the config provided default.
	ListDefaulting PodTopologySpreadConstraintsDefaulting = "List"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// PodTopologySpreadArgs holds arguments used to configure the PodTopologySpread plugin.
type PodTopologySpreadArgs struct {
	metav1.TypeMeta `json:",inline"`

	// DefaultConstraints defines topology spread constraints to be applied to
	// Pods that don't define any in `pod.spec.topologySpreadConstraints`.
	// `.defaultConstraints[*].labelSelectors` must be empty, as they are
	// deduced from the Pod's membership to Services, ReplicationControllers,
	// ReplicaSets or StatefulSets.
	// When not empty, .defaultingType must be "List".
	// +optional
	// +listType=atomic
	DefaultConstraints []corev1.TopologySpreadConstraint `json:"defaultConstraints,omitempty"`

	// DefaultingType determines how .defaultConstraints are deduced. Can be one
	// of "System" or "List".
	//
	// - "System": Use kubernetes defined constraints that spread Pods among
	//   Nodes and Zones.
	// - "List": Use constraints defined in .defaultConstraints.
	//
	// Defaults to "System".
	// +optional
	DefaultingType PodTopologySpreadConstraintsDefaulting `json:"defaultingType,omitempty"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// NodeResourcesBalancedAllocationArgs holds arguments used to configure NodeResourcesBalancedAllocation plugin.
type NodeResourcesBalancedAllocationArgs struct {
	metav1.TypeMeta `json:",inline"`

	// Resources to be managed, the default is "cpu" and "memory" if not specified.
	// +listType=map
	// +listMapKey=name
	Resources []ResourceSpec `json:"resources,omitempty"`
}

// UtilizationShapePoint represents single point of priority function shape.
type UtilizationShapePoint struct {
	// Utilization (x axis). Valid values are 0 to 100. Fully utilized node maps to 100.
	Utilization int32 `json:"utilization"`
	// Score assigned to given utilization (y axis). Valid values are 0 to 10.
	Score int32 `json:"score"`
}

// ResourceSpec represents a single resource.
type ResourceSpec struct {
	// Name of the resource.
	Name string `json:"name"`
	// Weight of the resource.
	Weight int64 `json:"weight,omitempty"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// VolumeBindingArgs holds arguments used to configure the VolumeBinding plugin.
type VolumeBindingArgs struct {
	metav1.TypeMeta `json:",inline"`

	// BindTimeoutSeconds is the timeout in seconds in volume binding operation.
	// Value must be non-negative integer. The value zero indicates no waiting.
	// If this value is nil, the default value (600) will be used.
	BindTimeoutSeconds *int64 `json:"bindTimeoutSeconds,omitempty"`

	// Shape specifies the points defining the score function shape, which is
	// used to score nodes based on the utilization of statically provisioned
	// PVs. The utilization is calculated by dividing the total requested
	// storage of the pod by the total capacity of feasible PVs on each node.
	// Each point contains utilization (ranges from 0 to 100) and its
	// associated score (ranges from 0 to 10). You can turn the priority by
	// specifying different scores for different utilization numbers.
	// The default shape points are:
	// 1) 0 for 0 utilization
	// 2) 10 for 100 utilization
	// All points must be sorted in increasing order by utilization.
	// +featureGate=VolumeCapacityPriority
	// +optional
	// +listType=atomic
	Shape []UtilizationShapePoint `json:"shape,omitempty"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// NodeAffinityArgs holds arguments to configure the NodeAffinity plugin.
type NodeAffinityArgs struct {
	metav1.TypeMeta `json:",inline"`

	// AddedAffinity is applied to all Pods additionally to the NodeAffinity
	// specified in the PodSpec. That is, Nodes need to satisfy AddedAffinity
	// AND .spec.NodeAffinity. AddedAffinity is empty by default (all Nodes
	// match).
	// When AddedAffinity is used, some Pods with affinity requirements that match
	// a specific Node (such as Daemonset Pods) might remain unschedulable.
	// +optional
	AddedAffinity *corev1.NodeAffinity `json:"addedAffinity,omitempty"`
}

// ScoringStrategyType the type of scoring strategy used in NodeResourcesFit plugin.
type ScoringStrategyType string

const (
	// LeastAllocated strategy prioritizes nodes with least allocated resources.
	LeastAllocated ScoringStrategyType = "LeastAllocated"
	// MostAllocated strategy prioritizes nodes with most allocated resources.
	MostAllocated ScoringStrategyType = "MostAllocated"
	// RequestedToCapacityRatio strategy allows specifying a custom shape function
	// to score nodes based on the request to capacity ratio.
	RequestedToCapacityRatio ScoringStrategyType = "RequestedToCapacityRatio"
)

// ScoringStrategy define ScoringStrategyType for node resource plugin
type ScoringStrategy struct {
	// Type selects which strategy to run.
	Type ScoringStrategyType `json:"type,omitempty"`

	// Resources to consider when scoring.
	// The default resource set includes "cpu" and "memory" with an equal weight.
	// Allowed weights go from 1 to 100.
	// Weight defaults to 1 if not specified or explicitly set to 0.
	// +listType=map
	// +listMapKey=topologyKey
	Resources []ResourceSpec `json:"resources,omitempty"`

	// Arguments specific to RequestedToCapacityRatio strategy.
	RequestedToCapacityRatio *RequestedToCapacityRatioParam `json:"requestedToCapacityRatio,omitempty"`
}

// RequestedToCapacityRatioParam define RequestedToCapacityRatio parameters
type RequestedToCapacityRatioParam struct {
	// Shape is a list of points defining the scoring function shape.
	// +listType=atomic
	Shape []UtilizationShapePoint `json:"shape,omitempty"`
}
