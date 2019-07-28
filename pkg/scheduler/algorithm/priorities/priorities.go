/*
Copyright 2018 The Kubernetes Authors.

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

package priorities

const (
	// EqualPriority defines the name of prioritizer function that gives an equal weight of one to all nodes.
	EqualPriority = "EqualPriority"
	// MostRequestedPriority defines the name of prioritizer function that gives used nodes higher priority.
	MostRequestedPriority = "MostRequestedPriority"
	// RequestedToCapacityRatioPriority defines the name of RequestedToCapacityRatioPriority.
	RequestedToCapacityRatioPriority = "RequestedToCapacityRatioPriority"
	// SelectorSpreadPriority defines the name of prioritizer function that spreads pods by minimizing
	// the number of pods (belonging to the same service or replication controller) on the same node.
	SelectorSpreadPriority = "SelectorSpreadPriority"
	// ServiceSpreadingPriority is largely replaced by "SelectorSpreadPriority".
	ServiceSpreadingPriority = "ServiceSpreadingPriority"
	// InterPodAffinityPriority defines the name of prioritizer function that decides which pods should or
	// should not be placed in the same topological domain as some other pods.
	InterPodAffinityPriority = "InterPodAffinityPriority"
	// LeastRequestedPriority defines the name of prioritizer function that prioritize nodes by least
	// requested utilization.
	LeastRequestedPriority = "LeastRequestedPriority"
	// BalancedResourceAllocation defines the name of prioritizer function that prioritizes nodes
	// to help achieve balanced resource usage.
	BalancedResourceAllocation = "BalancedResourceAllocation"
	// NodePreferAvoidPodsPriority defines the name of prioritizer function that priorities nodes according to
	// the node annotation "scheduler.alpha.kubernetes.io/preferAvoidPods".
	NodePreferAvoidPodsPriority = "NodePreferAvoidPodsPriority"
	// NodeAffinityPriority defines the name of prioritizer function that prioritizes nodes which have labels
	// matching NodeAffinity.
	NodeAffinityPriority = "NodeAffinityPriority"
	// TaintTolerationPriority defines the name of prioritizer function that prioritizes nodes that marked
	// with taint which pod can tolerate.
	TaintTolerationPriority = "TaintTolerationPriority"
	// ImageLocalityPriority defines the name of prioritizer function that prioritizes nodes that have images
	// requested by the pod present.
	ImageLocalityPriority = "ImageLocalityPriority"
	// ResourceLimitsPriority defines the nodes of prioritizer function ResourceLimitsPriority.
	ResourceLimitsPriority = "ResourceLimitsPriority"
	// EvenPodsSpreadPriority defines the name of prioritizer function that prioritizes nodes
	// which have pods and labels matching the incoming pod's topologySpreadConstraints.
	EvenPodsSpreadPriority = "EvenPodsSpreadPriority"
)
