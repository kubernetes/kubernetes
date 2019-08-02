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

package interpodaffinity

import (
	"sync"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	priorityutil "k8s.io/kubernetes/pkg/scheduler/algorithm/priorities/util"
)

type podAffinityPriorityMap struct {
	sync.Mutex

	// counts store the mapping from node name to so-far computed score of
	// the node.
	counts map[string]int64
}

func newPodAffinityPriorityMap() *podAffinityPriorityMap {
	return &podAffinityPriorityMap{
		counts: make(map[string]int64),
	}
}

func (p *podAffinityPriorityMap) processTerm(term *v1.PodAffinityTerm, podDefiningAffinityTerm, podToCheck *v1.Pod, fixedNode *v1.Node, nodes []*v1.Node, weight int64) error {
	namespaces := priorityutil.GetNamespacesFromPodAffinityTerm(podDefiningAffinityTerm, term)
	selector, err := metav1.LabelSelectorAsSelector(term.LabelSelector)
	if err != nil {
		return err
	}
	match := priorityutil.PodMatchesTermsNamespaceAndSelector(podToCheck, namespaces, selector)
	if match {
		for _, node := range nodes {
			if priorityutil.NodesHaveSameTopologyKey(node, fixedNode, term.TopologyKey) {
				p.Lock()
				p.counts[node.Name] += weight
				p.Unlock()
			}
		}
	}

	return nil
}

func (p *podAffinityPriorityMap) processTerms(terms []v1.WeightedPodAffinityTerm, podDefiningAffinityTerm, podToCheck *v1.Pod, fixedNode *v1.Node, nodes []*v1.Node, multiplier int) error {
	for i := range terms {
		term := &terms[i]
		if err := p.processTerm(&term.PodAffinityTerm, podDefiningAffinityTerm, podToCheck, fixedNode, nodes, int64(term.Weight*int32(multiplier))); err != nil {
			return err
		}
	}

	return nil
}
