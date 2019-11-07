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
	"context"
	"sync"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/util/workqueue"
	priorityutil "k8s.io/kubernetes/pkg/scheduler/algorithm/priorities/util"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	schedulerlisters "k8s.io/kubernetes/pkg/scheduler/listers"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"

	"k8s.io/klog"
)

// InterPodAffinity contains information to calculate inter pod affinity.
type InterPodAffinity struct {
	hardPodAffinityWeight int32
}

// NewInterPodAffinityPriority creates an InterPodAffinity.
func NewInterPodAffinityPriority(hardPodAffinityWeight int32) PriorityFunction {
	interPodAffinity := &InterPodAffinity{
		hardPodAffinityWeight: hardPodAffinityWeight,
	}
	return interPodAffinity.CalculateInterPodAffinityPriority
}

type topologyPairToScore map[string]map[string]int64

type podAffinityPriorityMap struct {
	// nodes contain all nodes that should be considered.
	nodes []*v1.Node
	// tracks a topology pair score so far.
	topologyScore topologyPairToScore
	sync.Mutex
}

func newPodAffinityPriorityMap(nodes []*v1.Node) *podAffinityPriorityMap {
	return &podAffinityPriorityMap{
		nodes:         nodes,
		topologyScore: make(topologyPairToScore),
	}
}

func (p *podAffinityPriorityMap) processTerm(term *v1.PodAffinityTerm, podDefiningAffinityTerm, podToCheck *v1.Pod, fixedNode *v1.Node, weight int64) error {
	namespaces := priorityutil.GetNamespacesFromPodAffinityTerm(podDefiningAffinityTerm, term)
	selector, err := metav1.LabelSelectorAsSelector(term.LabelSelector)
	if err != nil {
		return err
	}
	if len(fixedNode.Labels) == 0 {
		return nil
	}

	match := priorityutil.PodMatchesTermsNamespaceAndSelector(podToCheck, namespaces, selector)
	tpValue, tpValueExist := fixedNode.Labels[term.TopologyKey]
	if match && tpValueExist {
		p.Lock()
		if p.topologyScore[term.TopologyKey] == nil {
			p.topologyScore[term.TopologyKey] = make(map[string]int64)
		}
		p.topologyScore[term.TopologyKey][tpValue] += weight
		p.Unlock()
	}
	return nil
}

func (p *podAffinityPriorityMap) processTerms(terms []v1.WeightedPodAffinityTerm, podDefiningAffinityTerm, podToCheck *v1.Pod, fixedNode *v1.Node, multiplier int) error {
	for i := range terms {
		term := &terms[i]
		if err := p.processTerm(&term.PodAffinityTerm, podDefiningAffinityTerm, podToCheck, fixedNode, int64(term.Weight*int32(multiplier))); err != nil {
			return err
		}
	}
	return nil
}

// CalculateInterPodAffinityPriority compute a sum by iterating through the elements of weightedPodAffinityTerm and adding
// "weight" to the sum if the corresponding PodAffinityTerm is satisfied for
// that node; the node(s) with the highest sum are the most preferred.
// Symmetry need to be considered for preferredDuringSchedulingIgnoredDuringExecution from podAffinity & podAntiAffinity,
// symmetry need to be considered for hard requirements from podAffinity
func (ipa *InterPodAffinity) CalculateInterPodAffinityPriority(pod *v1.Pod, sharedLister schedulerlisters.SharedLister, nodes []*v1.Node) (framework.NodeScoreList, error) {
	affinity := pod.Spec.Affinity
	hasAffinityConstraints := affinity != nil && affinity.PodAffinity != nil
	hasAntiAffinityConstraints := affinity != nil && affinity.PodAntiAffinity != nil

	// pm stores (1) all nodes that should be considered and (2) the so-far computed score for each node.
	pm := newPodAffinityPriorityMap(nodes)

	allNodes, err := sharedLister.NodeInfos().HavePodsWithAffinityList()
	if err != nil {
		return nil, err
	}
	if hasAffinityConstraints || hasAntiAffinityConstraints {
		allNodes, err = sharedLister.NodeInfos().List()
		if err != nil {
			return nil, err
		}
	}

	// convert the topology key based weights to the node name based weights
	var maxCount, minCount int64

	processPod := func(existingPod *v1.Pod) error {
		existingPodNodeInfo, err := sharedLister.NodeInfos().Get(existingPod.Spec.NodeName)
		if err != nil {
			klog.Errorf("Node not found, %v", existingPod.Spec.NodeName)
			return nil
		}
		existingPodAffinity := existingPod.Spec.Affinity
		existingHasAffinityConstraints := existingPodAffinity != nil && existingPodAffinity.PodAffinity != nil
		existingHasAntiAffinityConstraints := existingPodAffinity != nil && existingPodAffinity.PodAntiAffinity != nil
		existingPodNode := existingPodNodeInfo.Node()

		if hasAffinityConstraints {
			// For every soft pod affinity term of <pod>, if <existingPod> matches the term,
			// increment <pm.counts> for every node in the cluster with the same <term.TopologyKey>
			// value as that of <existingPods>`s node by the term`s weight.
			terms := affinity.PodAffinity.PreferredDuringSchedulingIgnoredDuringExecution
			if err := pm.processTerms(terms, pod, existingPod, existingPodNode, 1); err != nil {
				return err
			}
		}
		if hasAntiAffinityConstraints {
			// For every soft pod anti-affinity term of <pod>, if <existingPod> matches the term,
			// decrement <pm.counts> for every node in the cluster with the same <term.TopologyKey>
			// value as that of <existingPod>`s node by the term`s weight.
			terms := affinity.PodAntiAffinity.PreferredDuringSchedulingIgnoredDuringExecution
			if err := pm.processTerms(terms, pod, existingPod, existingPodNode, -1); err != nil {
				return err
			}
		}

		if existingHasAffinityConstraints {
			// For every hard pod affinity term of <existingPod>, if <pod> matches the term,
			// increment <pm.counts> for every node in the cluster with the same <term.TopologyKey>
			// value as that of <existingPod>'s node by the constant <ipa.hardPodAffinityWeight>
			if ipa.hardPodAffinityWeight > 0 {
				terms := existingPodAffinity.PodAffinity.RequiredDuringSchedulingIgnoredDuringExecution
				// TODO: Uncomment this block when implement RequiredDuringSchedulingRequiredDuringExecution.
				//if len(existingPodAffinity.PodAffinity.RequiredDuringSchedulingRequiredDuringExecution) != 0 {
				//	terms = append(terms, existingPodAffinity.PodAffinity.RequiredDuringSchedulingRequiredDuringExecution...)
				//}
				for _, term := range terms {
					if err := pm.processTerm(&term, existingPod, pod, existingPodNode, int64(ipa.hardPodAffinityWeight)); err != nil {
						return err
					}
				}
			}
			// For every soft pod affinity term of <existingPod>, if <pod> matches the term,
			// increment <pm.counts> for every node in the cluster with the same <term.TopologyKey>
			// value as that of <existingPod>'s node by the term's weight.
			terms := existingPodAffinity.PodAffinity.PreferredDuringSchedulingIgnoredDuringExecution
			if err := pm.processTerms(terms, existingPod, pod, existingPodNode, 1); err != nil {
				return err
			}
		}
		if existingHasAntiAffinityConstraints {
			// For every soft pod anti-affinity term of <existingPod>, if <pod> matches the term,
			// decrement <pm.counts> for every node in the cluster with the same <term.TopologyKey>
			// value as that of <existingPod>'s node by the term's weight.
			terms := existingPodAffinity.PodAntiAffinity.PreferredDuringSchedulingIgnoredDuringExecution
			if err := pm.processTerms(terms, existingPod, pod, existingPodNode, -1); err != nil {
				return err
			}
		}
		return nil
	}

	errCh := schedutil.NewErrorChannel()
	ctx, cancel := context.WithCancel(context.Background())
	processNode := func(i int) {
		nodeInfo := allNodes[i]
		if nodeInfo.Node() != nil {
			if hasAffinityConstraints || hasAntiAffinityConstraints {
				// We need to process all the pods.
				for _, existingPod := range nodeInfo.Pods() {
					if err := processPod(existingPod); err != nil {
						errCh.SendErrorWithCancel(err, cancel)
						return
					}
				}
			} else {
				// The pod doesn't have any constraints - we need to check only existing
				// ones that have some.
				for _, existingPod := range nodeInfo.PodsWithAffinity() {
					if err := processPod(existingPod); err != nil {
						errCh.SendErrorWithCancel(err, cancel)
						return
					}
				}
			}
		}
	}
	workqueue.ParallelizeUntil(ctx, 16, len(allNodes), processNode)
	if err := errCh.ReceiveError(); err != nil {
		return nil, err
	}

	counts := make([]int64, len(nodes))
	for i := range nodes {
		if nodes[i].Labels != nil {
			for tpKey, tpValues := range pm.topologyScore {
				if v, exist := nodes[i].Labels[tpKey]; exist {
					counts[i] += tpValues[v]
				}
			}
		}
		if counts[i] > maxCount {
			maxCount = counts[i]
		}
		if counts[i] < minCount {
			minCount = counts[i]
		}
	}

	// calculate final priority score for each node
	result := make(framework.NodeScoreList, 0, len(nodes))
	maxMinDiff := maxCount - minCount
	for i, node := range nodes {
		fScore := float64(0)
		if maxMinDiff > 0 {
			fScore = float64(framework.MaxNodeScore) * (float64(counts[i]-minCount) / float64(maxCount-minCount))
		}
		result = append(result, framework.NodeScore{Name: node.Name, Score: int64(fScore)})
		if klog.V(10) {
			klog.Infof("%v -> %v: InterPodAffinityPriority, Score: (%d)", pod.Name, node.Name, int(fScore))
		}
	}
	return result, nil
}
