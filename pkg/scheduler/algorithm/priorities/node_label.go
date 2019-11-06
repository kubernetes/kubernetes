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

package priorities

import (
	"fmt"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
)

// NodeLabelPrioritizer contains information to calculate node label priority.
type NodeLabelPrioritizer struct {
	presentLabelsPreference []string
	absentLabelsPreference  []string
}

// NewNodeLabelPriority creates a NodeLabelPrioritizer.
func NewNodeLabelPriority(presentLabelsPreference []string, absentLabelsPreference []string) (PriorityMapFunction, PriorityReduceFunction) {
	labelPrioritizer := &NodeLabelPrioritizer{
		presentLabelsPreference: presentLabelsPreference,
		absentLabelsPreference:  absentLabelsPreference,
	}
	return labelPrioritizer.CalculateNodeLabelPriorityMap, nil
}

// CalculateNodeLabelPriorityMap checks whether a particular label exists on a node or not, regardless of its value.
// If presence is true, prioritizes nodes that have the specified label, regardless of value.
// If presence is false, prioritizes nodes that do not have the specified label.
func (n *NodeLabelPrioritizer) CalculateNodeLabelPriorityMap(pod *v1.Pod, meta interface{}, nodeInfo *schedulernodeinfo.NodeInfo) (framework.NodeScore, error) {
	node := nodeInfo.Node()
	if node == nil {
		return framework.NodeScore{}, fmt.Errorf("node not found")
	}

	score := int64(0)
	for _, label := range n.presentLabelsPreference {
		if labels.Set(node.Labels).Has(label) {
			score += framework.MaxNodeScore
		}
	}
	for _, label := range n.absentLabelsPreference {
		if !labels.Set(node.Labels).Has(label) {
			score += framework.MaxNodeScore
		}
	}
	// Take average score for each label to ensure the score doesn't exceed MaxNodeScore.
	score /= int64(len(n.presentLabelsPreference) + len(n.absentLabelsPreference))

	return framework.NodeScore{
		Name:  node.Name,
		Score: score,
	}, nil
}
