/*
Copyright 2015 The Kubernetes Authors.

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
	"k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

// NodeInfo is node level aggregated information.
type NodeInfo = framework.NodeInfo

// NewNodeInfo returns a ready to use empty NodeInfo object.
// If any pods are given in arguments, their information will be aggregated in
// the returned object.
func NewNodeInfo(pods ...*v1.Pod) *NodeInfo {
	return framework.NewNodeInfo(pods...)
}

// NewPodInfo return a new PodInfo
func NewPodInfo(pod *v1.Pod) *PodInfo {
	return framework.NewPodInfo(pod)
}

// Resource is a collection of compute resource.
type Resource = framework.Resource

// QueuedPodInfo is a Pod wrapper with additional information related to
// the pod's status in the scheduling queue, such as the timestamp when
// it's added to the queue.
type QueuedPodInfo = framework.QueuedPodInfo

// PodInfo is a wrapper to a Pod with additional pre-computed information to
// accelerate processing. This information is typically immutable (e.g., pre-processed
// inter-pod affinity selectors).
type PodInfo = framework.PodInfo

// AffinityTerm is a processed version of v1.PodAffinityTerm.
type AffinityTerm = framework.AffinityTerm

// WeightedAffinityTerm is a "processed" representation of v1.WeightedAffinityTerm.
type WeightedAffinityTerm = framework.WeightedAffinityTerm

// ImageStateSummary provides summarized information about the state of an image.
type ImageStateSummary = framework.ImageStateSummary

// TransientSchedulerInfo is a transient structure which is destructed at the end of each scheduling cycle.
// It consists of items that are valid for a scheduling cycle and is used for message passing across predicates and
// priorities. Some examples which could be used as fields are number of volumes being used on node, current utilization
// on node etc.
// IMPORTANT NOTE: Make sure that each field in this structure is documented along with usage. Expand this structure
// only when absolutely needed as this data structure will be created and destroyed during every scheduling cycle.
type TransientSchedulerInfo = framework.TransientSchedulerInfo

// HostPortInfo stores mapping from ip to a set of ProtocolPort
type HostPortInfo = framework.HostPortInfo

// ProtocolPort represents a protocol port pair, e.g. tcp:80.
type ProtocolPort = framework.ProtocolPort

// NewResource creates a Resource from ResourceList
func NewResource(rl v1.ResourceList) *Resource {
	return framework.NewResource(rl)
}
