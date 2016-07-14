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
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm/predicates"
	priorityutil "k8s.io/kubernetes/plugin/pkg/scheduler/algorithm/priorities/util"
	schedulerapi "k8s.io/kubernetes/plugin/pkg/scheduler/api"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"
)

type InterPodAffinity struct {
	info                  predicates.NodeInfo
	nodeLister            algorithm.NodeLister
	podLister             algorithm.PodLister
	hardPodAffinityWeight int
	failureDomains        priorityutil.Topologies
}

func NewInterPodAffinityPriority(
	info predicates.NodeInfo,
	nodeLister algorithm.NodeLister,
	podLister algorithm.PodLister,
	hardPodAffinityWeight int,
	failureDomains []string) algorithm.PriorityFunction {
	interPodAffinity := &InterPodAffinity{
		info:                  info,
		nodeLister:            nodeLister,
		podLister:             podLister,
		hardPodAffinityWeight: hardPodAffinityWeight,
		failureDomains:        priorityutil.Topologies{DefaultKeys: failureDomains},
	}
	return interPodAffinity.CalculateInterPodAffinityPriority
}

// TODO: Share it with predicates by moving to better location.
// TODO: Can we avoid error handling here - this is only a matter of non-parsable selector?
func podMatchesNamespaceAndSelector(pod *api.Pod, affinityPod *api.Pod, term *api.PodAffinityTerm) (bool, error) {
	namespaces := priorityutil.GetNamespacesFromPodAffinityTerm(affinityPod, *term)
	if len(namespaces) != 0 && !namespaces.Has(pod.Namespace) {
		return false, nil
	}

	selector, err := unversioned.LabelSelectorAsSelector(term.LabelSelector)
	if err != nil || !selector.Matches(labels.Set(pod.Labels)) {
		return false, err
	}
	return true, nil
}

// compute a sum by iterating through the elements of weightedPodAffinityTerm and adding
// "weight" to the sum if the corresponding PodAffinityTerm is satisfied for
// that node; the node(s) with the highest sum are the most preferred.
// Symmetry need to be considered for preferredDuringSchedulingIgnoredDuringExecution from podAffinity & podAntiAffinity,
// symmetry need to be considered for hard requirements from podAffinity
func (ipa *InterPodAffinity) CalculateInterPodAffinityPriority(pod *api.Pod, nodeNameToInfo map[string]*schedulercache.NodeInfo, nodeLister algorithm.NodeLister) (schedulerapi.HostPriorityList, error) {
	nodes, err := nodeLister.List()
	if err != nil {
		return nil, err
	}
	allPods, err := ipa.podLister.List(labels.Everything())
	if err != nil {
		return nil, err
	}
	affinity, err := api.GetAffinityFromPodAnnotations(pod.Annotations)
	if err != nil {
		return nil, err
	}

	// convert the topology key based weights to the node name based weights
	var maxCount float64
	var minCount float64
	// counts store the mapping from node name to so-far computed score of
	// the node.
	counts := make(map[string]float64, len(nodes))

	processTerm := func(term *api.PodAffinityTerm, affinityPod, podToCheck *api.Pod, fixedNode *api.Node, weight float64) error {
		match, err := podMatchesNamespaceAndSelector(podToCheck, affinityPod, term)
		if err != nil {
			return err
		}
		if match {
			for _, node := range nodes {
				if ipa.failureDomains.NodesHaveSameTopologyKey(node, fixedNode, term.TopologyKey) {
					counts[node.Name] += weight
				}
			}
		}
		return nil
	}
	processTerms := func(terms []api.WeightedPodAffinityTerm, affinityPod, podToCheck *api.Pod, fixedNode *api.Node, multiplier int) error {
		for _, weightedTerm := range terms {
			if err := processTerm(&weightedTerm.PodAffinityTerm, affinityPod, podToCheck, fixedNode, float64(weightedTerm.Weight*multiplier)); err != nil {
				return err
			}
		}
		return nil
	}

	for _, existingPod := range allPods {
		existingPodNode, err := ipa.info.GetNodeInfo(existingPod.Spec.NodeName)
		if err != nil {
			return nil, err
		}
		existingPodAffinity, err := api.GetAffinityFromPodAnnotations(existingPod.Annotations)
		if err != nil {
			return nil, err
		}

		if affinity.PodAffinity != nil {
			// For every soft pod affinity term of <pod>, if <existingPod> matches the term,
			// increment <counts> for every node in the cluster with the same <term.TopologyKey>
			// value as that of <existingPods>`s node by the term`s weight.
			terms := affinity.PodAffinity.PreferredDuringSchedulingIgnoredDuringExecution
			if err := processTerms(terms, pod, existingPod, existingPodNode, 1); err != nil {
				return nil, err
			}
		}
		if affinity.PodAntiAffinity != nil {
			// For every soft pod anti-affinity term of <pod>, if <existingPod> matches the term,
			// decrement <counts> for every node in the cluster with the same <term.TopologyKey>
			// value as that of <existingPod>`s node by the term`s weight.
			terms := affinity.PodAntiAffinity.PreferredDuringSchedulingIgnoredDuringExecution
			if err := processTerms(terms, pod, existingPod, existingPodNode, -1); err != nil {
				return nil, err
			}
		}

		if existingPodAffinity.PodAffinity != nil {
			// For every hard pod affinity term of <existingPod>, if <pod> matches the term,
			// increment <counts> for every node in the cluster with the same <term.TopologyKey>
			// value as that of <existingPod>'s node by the constant <ipa.hardPodAffinityWeight>
			if ipa.hardPodAffinityWeight > 0 {
				terms := existingPodAffinity.PodAffinity.RequiredDuringSchedulingIgnoredDuringExecution
				// TODO: Uncomment this block when implement RequiredDuringSchedulingRequiredDuringExecution.
				//if len(existingPodAffinity.PodAffinity.RequiredDuringSchedulingRequiredDuringExecution) != 0 {
				//	terms = append(terms, existingPodAffinity.PodAffinity.RequiredDuringSchedulingRequiredDuringExecution...)
				//}
				for _, term := range terms {
					if err := processTerm(&term, existingPod, pod, existingPodNode, float64(ipa.hardPodAffinityWeight)); err != nil {
						return nil, err
					}
				}
			}
			// For every soft pod affinity term of <existingPod>, if <pod> matches the term,
			// increment <counts> for every node in the cluster with the same <term.TopologyKey>
			// value as that of <existingPod>'s node by the term's weight.
			terms := existingPodAffinity.PodAffinity.PreferredDuringSchedulingIgnoredDuringExecution
			if err := processTerms(terms, existingPod, pod, existingPodNode, 1); err != nil {
				return nil, err
			}
		}
		if existingPodAffinity.PodAntiAffinity != nil {
			// For every soft pod anti-affinity term of <existingPod>, if <pod> matches the term,
			// decrement <counts> for every node in the cluster with the same <term.TopologyKey>
			// value as that of <existingPod>'s node by the term's weight.
			terms := existingPodAffinity.PodAntiAffinity.PreferredDuringSchedulingIgnoredDuringExecution
			if err := processTerms(terms, existingPod, pod, existingPodNode, -1); err != nil {
				return nil, err
			}
		}
	}

	for _, node := range nodes {
		if counts[node.Name] > maxCount {
			maxCount = counts[node.Name]
		}
		if counts[node.Name] < minCount {
			minCount = counts[node.Name]
		}
	}

	// calculate final priority score for each node
	result := make(schedulerapi.HostPriorityList, 0, len(nodes))
	for _, node := range nodes {
		fScore := float64(0)
		if (maxCount - minCount) > 0 {
			fScore = 10 * ((counts[node.Name] - minCount) / (maxCount - minCount))
		}
		result = append(result, schedulerapi.HostPriority{Host: node.Name, Score: int(fScore)})
		if glog.V(10) {
			// We explicitly don't do glog.V(10).Infof() to avoid computing all the parameters if this is
			// not logged. There is visible performance gain from it.
			glog.V(10).Infof("%v -> %v: InterPodAffinityPriority, Score: (%d)", pod.Name, node.Name, int(fScore))
		}
	}
	return result, nil
}
