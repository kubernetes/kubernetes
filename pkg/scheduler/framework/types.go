/*
Copyright 2019 The Kubernetes Authors.

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

package framework

import (
	"k8s.io/api/core/v1"
	schedulertypes "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
)

type NodeInfo = schedulertypes.NodeInfo

func NewNodeInfo(pods ...*v1.Pod) *NodeInfo {
	return schedulertypes.NewNodeInfo(pods...)
}

func NewPodInfo(pod *v1.Pod) *PodInfo {
	return schedulertypes.NewPodInfo(pod)
}

type Resource = schedulertypes.Resource

type QueuedPodInfo = schedulertypes.QueuedPodInfo

type PodInfo = schedulertypes.PodInfo

type AffinityTerm = schedulertypes.AffinityTerm

type WeightedAffinityTerm = schedulertypes.WeightedAffinityTerm

type ImageStateSummary = schedulertypes.ImageStateSummary

type TransientSchedulerInfo = schedulertypes.TransientSchedulerInfo

type HostPortInfo = schedulertypes.HostPortInfo

type ProtocolPort = schedulertypes.ProtocolPort

func NewResource(rl v1.ResourceList) *Resource {
	return schedulertypes.NewResource(rl)
}
