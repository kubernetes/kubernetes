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

package pod

import (
	v1 "k8s.io/api/core/v1"
)

// NodeSelection specifies where to run a pod, using a combination of fixed node name,
// node selector and/or affinity.
type NodeSelection struct {
	Name     string
	Selector map[string]string
	Affinity *v1.Affinity
}

// setNodeAffinityRequirement sets affinity with specified operator to nodeName to nodeSelection
func setNodeAffinityRequirement(nodeSelection *NodeSelection, operator v1.NodeSelectorOperator, nodeName string) {
	// Add node-anti-affinity.
	if nodeSelection.Affinity == nil {
		nodeSelection.Affinity = &v1.Affinity{}
	}
	if nodeSelection.Affinity.NodeAffinity == nil {
		nodeSelection.Affinity.NodeAffinity = &v1.NodeAffinity{}
	}
	if nodeSelection.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution == nil {
		nodeSelection.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution = &v1.NodeSelector{}
	}
	nodeSelection.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms = append(nodeSelection.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms,
		v1.NodeSelectorTerm{
			MatchFields: []v1.NodeSelectorRequirement{
				{Key: "metadata.name", Operator: operator, Values: []string{nodeName}},
			},
		})
}

// SetNodeAffinityTopologyRequirement sets node affinity to a specified topology
func SetNodeAffinityTopologyRequirement(nodeSelection *NodeSelection, topology map[string]string) {
	if nodeSelection.Affinity == nil {
		nodeSelection.Affinity = &v1.Affinity{}
	}
	if nodeSelection.Affinity.NodeAffinity == nil {
		nodeSelection.Affinity.NodeAffinity = &v1.NodeAffinity{}
	}
	if nodeSelection.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution == nil {
		nodeSelection.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution = &v1.NodeSelector{}
	}
	for k, v := range topology {
		nodeSelection.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms = append(nodeSelection.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms,
			v1.NodeSelectorTerm{
				MatchExpressions: []v1.NodeSelectorRequirement{
					{Key: k, Operator: v1.NodeSelectorOpIn, Values: []string{v}},
				},
			})

	}
}

// SetAffinity sets affinity to nodeName to nodeSelection
func SetAffinity(nodeSelection *NodeSelection, nodeName string) {
	setNodeAffinityRequirement(nodeSelection, v1.NodeSelectorOpIn, nodeName)
}

// SetAntiAffinity sets anti-affinity to nodeName to nodeSelection
func SetAntiAffinity(nodeSelection *NodeSelection, nodeName string) {
	setNodeAffinityRequirement(nodeSelection, v1.NodeSelectorOpNotIn, nodeName)
}

// SetNodeAffinity modifies the given pod object with
// NodeAffinity to the given node name.
func SetNodeAffinity(podSpec *v1.PodSpec, nodeName string) {
	nodeSelection := &NodeSelection{}
	SetAffinity(nodeSelection, nodeName)
	podSpec.Affinity = nodeSelection.Affinity
}

// SetNodeSelection modifies the given pod object with
// the specified NodeSelection
func SetNodeSelection(podSpec *v1.PodSpec, nodeSelection NodeSelection) {
	podSpec.NodeSelector = nodeSelection.Selector
	podSpec.Affinity = nodeSelection.Affinity
	// pod.Spec.NodeName should not be set directly because
	// it will bypass the scheduler, potentially causing
	// kubelet to Fail the pod immediately if it's out of
	// resources. Instead, we want the pod to remain
	// pending in the scheduler until the node has resources
	// freed up.
	if nodeSelection.Name != "" {
		SetNodeAffinity(podSpec, nodeSelection.Name)
	}
}
