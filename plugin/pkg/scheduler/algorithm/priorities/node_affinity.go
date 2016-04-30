/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
	schedulerapi "k8s.io/kubernetes/plugin/pkg/scheduler/api"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"
)

type NodeAffinity struct {
	nodeLister algorithm.NodeLister
}

func NewNodeAffinityPriority(nodeLister algorithm.NodeLister) algorithm.PriorityFunction {
	nodeAffinity := &NodeAffinity{
		nodeLister: nodeLister,
	}
	return nodeAffinity.CalculateNodeAffinityPriority
}

// CalculateNodeAffinityPriority prioritizes nodes according to node affinity scheduling preferences
// indicated in PreferredDuringSchedulingIgnoredDuringExecution. Each time a node match a preferredSchedulingTerm,
// it will a get an add of preferredSchedulingTerm.Weight. Thus, the more preferredSchedulingTerms
// the node satisfies and the more the preferredSchedulingTerm that is satisfied weights, the higher
// score the node gets.
func (s *NodeAffinity) CalculateNodeAffinityPriority(pod *api.Pod, nodeNameToInfo map[string]*schedulercache.NodeInfo, nodeLister algorithm.NodeLister) (schedulerapi.HostPriorityList, error) {
	var maxCount int
	counts := map[string]int{}

	nodes, err := nodeLister.List()
	if err != nil {
		return nil, err
	}

	affinity, err := api.GetAffinityFromPodAnnotations(pod.Annotations)
	if err != nil {
		return nil, err
	}

	// A nil element of PreferredDuringSchedulingIgnoredDuringExecution matches no objects.
	// An element of PreferredDuringSchedulingIgnoredDuringExecution that refers to an
	// empty PreferredSchedulingTerm matches all objects.
	if affinity.NodeAffinity != nil && affinity.NodeAffinity.PreferredDuringSchedulingIgnoredDuringExecution != nil {
		// Match PreferredDuringSchedulingIgnoredDuringExecution term by term.
		for _, preferredSchedulingTerm := range affinity.NodeAffinity.PreferredDuringSchedulingIgnoredDuringExecution {
			if preferredSchedulingTerm.Weight == 0 {
				continue
			}

			nodeSelector, err := api.NodeSelectorRequirementsAsSelector(preferredSchedulingTerm.Preference.MatchExpressions)
			if err != nil {
				return nil, err
			}

			for _, node := range nodes.Items {
				if nodeSelector.Matches(labels.Set(node.Labels)) {
					counts[node.Name] += int(preferredSchedulingTerm.Weight)
				}

				if counts[node.Name] > maxCount {
					maxCount = counts[node.Name]
				}
			}
		}
	}

	result := []schedulerapi.HostPriority{}
	for _, node := range nodes.Items {
		fScore := float64(0)
		if maxCount > 0 {
			fScore = 10 * (float64(counts[node.Name]) / float64(maxCount))
		}
		result = append(result, schedulerapi.HostPriority{Host: node.Name, Score: int(fScore)})
		glog.V(10).Infof("%v -> %v: NodeAffinityPriority, Score: (%d)", pod.Name, node.Name, int(fScore))
	}
	return result, nil
}
