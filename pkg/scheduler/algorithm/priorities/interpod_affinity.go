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
	"fmt"
	"sync"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/util/workqueue"
	priorityutil "k8s.io/kubernetes/pkg/scheduler/algorithm/priorities/util"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	schedulerlisters "k8s.io/kubernetes/pkg/scheduler/listers"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"

	"k8s.io/klog"
)

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

// CalculateInterPodAffinityPriorityMap calculate the number of matching pods on the passed-in "node",
// and return the number as Score.
func CalculateInterPodAffinityPriorityMap(pod *v1.Pod, meta interface{}, nodeInfo *schedulernodeinfo.NodeInfo) (framework.NodeScore, error) {
	node := nodeInfo.Node()
	if node == nil {
		return framework.NodeScore{}, fmt.Errorf("node not found")
	}

	var topologyScore topologyPairToScore
	if priorityMeta, ok := meta.(*priorityMetadata); ok {
		topologyScore = priorityMeta.topologyScore
	}

	var score int64
	for tpKey, tpValues := range topologyScore {
		if v, exist := node.Labels[tpKey]; exist {
			score += tpValues[v]
		}
	}

	return framework.NodeScore{Name: node.Name, Score: score}, nil
}

// CalculateInterPodAffinityPriorityReduce normalizes the score for each filteredNode,
// The basic rule is: the bigger the score(matching number of pods) is, the smaller the
// final normalized score will be.
func CalculateInterPodAffinityPriorityReduce(pod *v1.Pod, meta interface{}, sharedLister schedulerlisters.SharedLister,
	result framework.NodeScoreList) error {
	var topologyScore topologyPairToScore
	if priorityMeta, ok := meta.(*priorityMetadata); ok {
		topologyScore = priorityMeta.topologyScore
	}
	if len(topologyScore) == 0 {
		return nil
	}

	var maxCount, minCount int64
	for i := range result {
		score := result[i].Score
		if score > maxCount {
			maxCount = score
		}
		if score < minCount {
			minCount = score
		}
	}

	maxMinDiff := maxCount - minCount
	for i := range result {
		fScore := float64(0)
		if maxMinDiff > 0 {
			fScore = float64(framework.MaxNodeScore) * (float64(result[i].Score-minCount) / float64(maxMinDiff))
		}

		result[i].Score = int64(fScore)
	}

	return nil
}

func buildTopologyPairToScore(
	pod *v1.Pod,
	sharedLister schedulerlisters.SharedLister,
	filteredNodes []*v1.Node,
	hardPodAffinityWeight int32,
) topologyPairToScore {
	if sharedLister == nil {
		klog.Error("BuildTopologyPairToScore with empty shared lister")
		return nil
	}

	affinity := pod.Spec.Affinity
	hasAffinityConstraints := affinity != nil && affinity.PodAffinity != nil
	hasAntiAffinityConstraints := affinity != nil && affinity.PodAntiAffinity != nil

	// pm stores (1) all nodes that should be considered and (2) the so-far computed score for each node.
	pm := newPodAffinityPriorityMap(filteredNodes)

	allNodes, err := sharedLister.NodeInfos().HavePodsWithAffinityList()
	if err != nil {
		klog.Errorf("get pods with affinity list error, err: %v", err)
		return nil
	}
	if hasAffinityConstraints || hasAntiAffinityConstraints {
		allNodes, err = sharedLister.NodeInfos().List()
		if err != nil {
			klog.Errorf("get all nodes from shared lister error, err: %v", err)
			return nil
		}
	}

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
			if hardPodAffinityWeight > 0 {
				terms := existingPodAffinity.PodAffinity.RequiredDuringSchedulingIgnoredDuringExecution
				// TODO: Uncomment this block when implement RequiredDuringSchedulingRequiredDuringExecution.
				//if len(existingPodAffinity.PodAffinity.RequiredDuringSchedulingRequiredDuringExecution) != 0 {
				//	terms = append(terms, existingPodAffinity.PodAffinity.RequiredDuringSchedulingRequiredDuringExecution...)
				//}
				for _, term := range terms {
					if err := pm.processTerm(&term, existingPod, pod, existingPodNode, int64(hardPodAffinityWeight)); err != nil {
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
		klog.Error(err)
		return nil
	}

	return pm.topologyScore
}
