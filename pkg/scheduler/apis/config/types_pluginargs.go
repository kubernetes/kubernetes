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

package config

import (
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// DefaultPreemptionArgs holds arguments used to configure the
// DefaultPreemption plugin.
type DefaultPreemptionArgs struct {
	metav1.TypeMeta

	// MinCandidateNodesPercentage is the minimum number of candidates to
	// shortlist when dry running preemption as a percentage of number of nodes.
	// Must be in the range [0, 100]. Defaults to 10% of the cluster size if
	// unspecified.
	MinCandidateNodesPercentage int32
	// MinCandidateNodesAbsolute is the absolute minimum number of candidates to
	// shortlist. The likely number of candidates enumerated for dry running
	// preemption is given by the formula:
	// numCandidates = max(numNodes * minCandidateNodesPercentage, minCandidateNodesAbsolute)
	// We say "likely" because there are other factors such as PDB violations
	// that play a role in the number of candidates shortlisted. Must be at least
	// 0 nodes. Defaults to 100 nodes if unspecified.
	MinCandidateNodesAbsolute int32
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// InterPodAffinityArgs holds arguments used to configure the InterPodAffinity plugin.
type InterPodAffinityArgs struct {
	metav1.TypeMeta

	// HardPodAffinityWeight is the scoring weight for existing pods with a
	// matching hard affinity to the incoming pod.
	HardPodAffinityWeight int32

	// IgnorePreferredTermsOfExistingPods configures the scheduler to ignore existing pods' preferred affinity
	// rules when scoring candidate nodes, unless the incoming pod has inter-pod affinities.
	IgnorePreferredTermsOfExistingPods bool
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// NodeResourcesFitArgs holds arguments used to configure the NodeResourcesFit plugin.
type NodeResourcesFitArgs struct {
	metav1.TypeMeta

	// IgnoredResources is the list of resources that NodeResources fit filter
	// should ignore.
	IgnoredResources []string
	// IgnoredResourceGroups defines the list of resource groups that NodeResources fit filter should ignore.
	// e.g. if group is ["example.com"], it will ignore all resource names that begin
	// with "example.com", such as "example.com/aaa" and "example.com/bbb".
	// A resource group name can't contain '/'.
	IgnoredResourceGroups []string

	// ScoringStrategy selects the node resource scoring strategy.
	ScoringStrategy *ScoringStrategy
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
	metav1.TypeMeta

	// DefaultConstraints defines topology spread constraints to be applied to
	// Pods that don't define any in `pod.spec.topologySpreadConstraints`.
	// `.defaultConstraints[*].labelSelectors` must be empty, as they are
	// deduced from the Pod's membership to Services, ReplicationControllers,
	// ReplicaSets or StatefulSets.
	// When not empty, .defaultingType must be "List".
	DefaultConstraints []v1.TopologySpreadConstraint

	// DefaultingType determines how .defaultConstraints are deduced. Can be one
	// of "System" or "List".
	//
	// - "System": Use kubernetes defined constraints that spread Pods among
	//   Nodes and Zones.
	// - "List": Use constraints defined in .defaultConstraints.
	//
	// Defaults to "System".
	// +optional
	DefaultingType PodTopologySpreadConstraintsDefaulting
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// NodeResourcesBalancedAllocationArgs holds arguments used to configure NodeResourcesBalancedAllocation plugin.
type NodeResourcesBalancedAllocationArgs struct {
	metav1.TypeMeta

	// Resources to be considered when scoring.
	// The default resource set includes "cpu" and "memory", only valid weight is 1.
	Resources []ResourceSpec
}

// UtilizationShapePoint represents a single point of a priority function shape.
type UtilizationShapePoint struct {
	// Utilization (x axis). Valid values are 0 to 100. Fully utilized node maps to 100.
	Utilization int32
	// Score assigned to a given utilization (y axis). Valid values are 0 to 10.
	Score int32
}

// ResourceSpec represents single resource.
type ResourceSpec struct {
	// Name of the resource.
	Name string
	// Weight of the resource.
	Weight int64
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// VolumeBindingArgs holds arguments used to configure the VolumeBinding plugin.
type VolumeBindingArgs struct {
	metav1.TypeMeta

	// BindTimeoutSeconds is the timeout in seconds in volume binding operation.
	// Value must be non-negative integer. The value zero indicates no waiting.
	// If this value is nil, the default value will be used.
	BindTimeoutSeconds int64

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
	// +featureGate=StorageCapacityScoring
	// +optional
	Shape []UtilizationShapePoint
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// NodeAffinityArgs holds arguments to configure the NodeAffinity plugin.
type NodeAffinityArgs struct {
	metav1.TypeMeta

	// AddedAffinity is applied to all Pods additionally to the NodeAffinity
	// specified in the PodSpec. That is, Nodes need to satisfy AddedAffinity
	// AND .spec.NodeAffinity. AddedAffinity is empty by default (all Nodes
	// match).
	// When AddedAffinity is used, some Pods with affinity requirements that match
	// a specific Node (such as Daemonset Pods) might remain unschedulable.
	AddedAffinity *v1.NodeAffinity
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
	Type ScoringStrategyType

	// Resources to consider when scoring.
	// The default resource set includes "cpu" and "memory" with an equal weight.
	// Allowed weights go from 1 to 100.
	// Weight defaults to 1 if not specified or explicitly set to 0.
	Resources []ResourceSpec

	// Arguments specific to RequestedToCapacityRatio strategy.
	RequestedToCapacityRatio *RequestedToCapacityRatioParam
}

// RequestedToCapacityRatioParam define RequestedToCapacityRatio parameters
type RequestedToCapacityRatioParam struct {
	// Shape is a list of points defining the scoring function shape.
	Shape []UtilizationShapePoint
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// DynamicResourcesArgs holds arguments used to configure the DynamicResources plugin.
type DynamicResourcesArgs struct {
	metav1.TypeMeta

	// FilterTimeout limits the amount of time that the filter operation may
	// take per node to search for devices that can be allocated to scheduler
	// a pod to that node.
	//
	// In typical scenarios, this operation should complete in 10 to 200
	// milliseconds, but could also be longer depending on the number of
	// requests per ResourceClaim, number of ResourceClaims, number of
	// published devices in ResourceSlices, and the complexity of the
	// requests. Other checks besides CEL evaluation also take time (usage
	// checks, match attributes, etc.).
	//
	// Therefore the scheduler plugin applies this timeout. If the timeout
	// is reached, the Pod is considered unschedulable for the node.
	// If filtering succeeds for some other node(s), those are picked instead.
	// If filtering fails for all of them, the Pod is placed in the
	// unschedulable queue. It will get checked again if changes in
	// e.g. ResourceSlices or ResourceClaims indicate that
	// another scheduling attempt might succeed. If this fails repeatedly,
	// exponential backoff slows down future attempts.
	//
	// The default is 10 seconds.
	// This is sufficient to prevent worst-case scenarios while not impacting normal
	// usage of DRA. However, slow filtering can slow down Pod scheduling
	// also for Pods not using DRA. Administators can reduce the timeout
	// after checking the
	// `scheduler_plugin_execution_duration_seconds` metrics.
	// That tracks the time spend in each Filter operation.
	// There's also `scheduler_framework_extension_point_duration_seconds`
	// which tracks the duration of filtering overall.
	//
	// Setting it to zero completely disables the timeout.
	FilterTimeout *metav1.Duration

	// BindingTimeout limits how long the PreBind extension point may wait for
	// ResourceClaim device BindingConditions to become satisfied when such
	// conditions are present. While waiting, the scheduler periodically checks
	// device status. If the timeout elapses before all required conditions are
	// true (or any bindingFailureConditions become true), the allocation is
	// cleared and the Pod re-enters scheduling queue. Note that the same or other node may be
	// chosen if feasible; otherwise the Pod is placed in the unschedulable queue and
	// retried based on cluster changes and backoff.
	//
	// Defaults & feature gates:
	//   - Defaults to 10 minutes when the DRADeviceBindingConditions feature gate is enabled.
	//   - Has effect only when BOTH DRADeviceBindingConditions and
	//     DRAResourceClaimDeviceStatus are enabled; otherwise omit this field.
	//   - When DRADeviceBindingConditions is disabled, setting this field is considered an error.
	//
	// Valid values:
	//   - >=1s (non-zero). No upper bound is enforced.
	//
	// Tuning guidance:
	//   - Lower values reduce time-to-retry when devices aren’t ready but can
	//     increase churn if drivers typically need longer to report readiness.
	//   - Review scheduler latency metrics (e.g. PreBind duration in
	//     `scheduler_framework_extension_point_duration_seconds`) and driver
	//     readiness behavior before tightening this timeout.
	BindingTimeout *metav1.Duration
}

const DynamicResourcesFilterTimeoutDefault = 10 * time.Second
const DynamicResourcesBindingTimeoutDefault = 600 * time.Second
