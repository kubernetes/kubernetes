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
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/sets"
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
	topologyScore         topologyPairToScore
	affinityTerms         []*weightedAffinityTerm
	antiAffinityTerms     []*weightedAffinityTerm
	hardPodAffinityWeight int32
	sync.Mutex
}

// A "processed" representation of v1.WeightedAffinityTerm.
type weightedAffinityTerm struct {
	namespaces  sets.String
	selector    labels.Selector
	weight      int32
	topologyKey string
}

func newWeightedAffinityTerm(pod *v1.Pod, term *v1.PodAffinityTerm, weight int32) (*weightedAffinityTerm, error) {
	namespaces := priorityutil.GetNamespacesFromPodAffinityTerm(pod, term)
	selector, err := metav1.LabelSelectorAsSelector(term.LabelSelector)
	if err != nil {
		return nil, err
	}
	return &weightedAffinityTerm{namespaces: namespaces, selector: selector, topologyKey: term.TopologyKey, weight: weight}, nil
}

func getProcessedTerms(pod *v1.Pod, terms []v1.WeightedPodAffinityTerm) ([]*weightedAffinityTerm, error) {
	if terms == nil {
		return nil, nil
	}

	var processedTerms []*weightedAffinityTerm
	for i := range terms {
		p, err := newWeightedAffinityTerm(pod, &terms[i].PodAffinityTerm, terms[i].Weight)
		if err != nil {
			return nil, err
		}
		processedTerms = append(processedTerms, p)
	}
	return processedTerms, nil
}

func (p *podAffinityPriorityMap) processTerm(term *weightedAffinityTerm, podToCheck *v1.Pod, fixedNode *v1.Node, multiplier int) error {
	if len(fixedNode.Labels) == 0 {
		return nil
	}

	match := priorityutil.PodMatchesTermsNamespaceAndSelector(podToCheck, term.namespaces, term.selector)
	tpValue, tpValueExist := fixedNode.Labels[term.topologyKey]
	if match && tpValueExist {
		p.Lock()
		if p.topologyScore[term.topologyKey] == nil {
			p.topologyScore[term.topologyKey] = make(map[string]int64)
		}
		p.topologyScore[term.topologyKey][tpValue] += int64(term.weight * int32(multiplier))
		p.Unlock()
	}
	return nil
}

func (p *podAffinityPriorityMap) processTerms(terms []*weightedAffinityTerm, podToCheck *v1.Pod, fixedNode *v1.Node, multiplier int) error {
	for _, term := range terms {
		if err := p.processTerm(term, podToCheck, fixedNode, multiplier); err != nil {
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

func (p *podAffinityPriorityMap) processExistingPod(existingPod *v1.Pod, existingPodNodeInfo *schedulernodeinfo.NodeInfo, incomingPod *v1.Pod) error {
	existingPodAffinity := existingPod.Spec.Affinity
	existingHasAffinityConstraints := existingPodAffinity != nil && existingPodAffinity.PodAffinity != nil
	existingHasAntiAffinityConstraints := existingPodAffinity != nil && existingPodAffinity.PodAntiAffinity != nil
	existingPodNode := existingPodNodeInfo.Node()

	// For every soft pod affinity term of <pod>, if <existingPod> matches the term,
	// increment <p.counts> for every node in the cluster with the same <term.TopologyKey>
	// value as that of <existingPods>`s node by the term`s weight.
	if err := p.processTerms(p.affinityTerms, existingPod, existingPodNode, 1); err != nil {
		return err
	}

	// For every soft pod anti-affinity term of <pod>, if <existingPod> matches the term,
	// decrement <p.counts> for every node in the cluster with the same <term.TopologyKey>
	// value as that of <existingPod>`s node by the term`s weight.
	if err := p.processTerms(p.antiAffinityTerms, existingPod, existingPodNode, -1); err != nil {
		return err
	}

	if existingHasAffinityConstraints {
		// For every hard pod affinity term of <existingPod>, if <pod> matches the term,
		// increment <p.counts> for every node in the cluster with the same <term.TopologyKey>
		// value as that of <existingPod>'s node by the constant <ipa.hardPodAffinityWeight>
		if p.hardPodAffinityWeight > 0 {
			terms := existingPodAffinity.PodAffinity.RequiredDuringSchedulingIgnoredDuringExecution
			// TODO: Uncomment this block when implement RequiredDuringSchedulingRequiredDuringExecution.
			//if len(existingPodAffinity.PodAffinity.RequiredDuringSchedulingRequiredDuringExecution) != 0 {
			//	terms = append(terms, existingPodAffinity.PodAffinity.RequiredDuringSchedulingRequiredDuringExecution...)
			//}
			for i := range terms {
				term := &terms[i]
				processedTerm, err := newWeightedAffinityTerm(existingPod, term, p.hardPodAffinityWeight)
				if err != nil {
					return err
				}
				if err := p.processTerm(processedTerm, incomingPod, existingPodNode, 1); err != nil {
					return err
				}
			}
		}
		// For every soft pod affinity term of <existingPod>, if <pod> matches the term,
		// increment <p.counts> for every node in the cluster with the same <term.TopologyKey>
		// value as that of <existingPod>'s node by the term's weight.
		terms, err := getProcessedTerms(existingPod, existingPodAffinity.PodAffinity.PreferredDuringSchedulingIgnoredDuringExecution)
		if err != nil {
			klog.Error(err)
			return nil
		}

		if err := p.processTerms(terms, incomingPod, existingPodNode, 1); err != nil {
			return err
		}
	}
	if existingHasAntiAffinityConstraints {
		// For every soft pod anti-affinity term of <existingPod>, if <pod> matches the term,
		// decrement <pm.counts> for every node in the cluster with the same <term.TopologyKey>
		// value as that of <existingPod>'s node by the term's weight.
		terms, err := getProcessedTerms(existingPod, existingPodAffinity.PodAntiAffinity.PreferredDuringSchedulingIgnoredDuringExecution)
		if err != nil {
			return err
		}
		if err := p.processTerms(terms, incomingPod, existingPodNode, -1); err != nil {
			return err
		}
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

	// Unless the pod being scheduled has affinity terms, we only
	// need to process nodes hosting pods with affinity.
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

	var affinityTerms []*weightedAffinityTerm
	var antiAffinityTerms []*weightedAffinityTerm
	if hasAffinityConstraints {
		if affinityTerms, err = getProcessedTerms(pod, affinity.PodAffinity.PreferredDuringSchedulingIgnoredDuringExecution); err != nil {
			klog.Error(err)
			return nil
		}
	}
	if hasAntiAffinityConstraints {
		if antiAffinityTerms, err = getProcessedTerms(pod, affinity.PodAntiAffinity.PreferredDuringSchedulingIgnoredDuringExecution); err != nil {
			klog.Error(err)
			return nil
		}
	}

	pm := podAffinityPriorityMap{
		topologyScore:         make(topologyPairToScore),
		affinityTerms:         affinityTerms,
		antiAffinityTerms:     antiAffinityTerms,
		hardPodAffinityWeight: hardPodAffinityWeight,
	}

	errCh := schedutil.NewErrorChannel()
	ctx, cancel := context.WithCancel(context.Background())
	processNode := func(i int) {
		nodeInfo := allNodes[i]
		if nodeInfo.Node() != nil {
			// Unless the pod being scheduled has affinity terms, we only
			// need to process pods with affinity in the node.
			podsToProcess := nodeInfo.PodsWithAffinity()
			if hasAffinityConstraints || hasAntiAffinityConstraints {
				// We need to process all the pods.
				podsToProcess = nodeInfo.Pods()
			}

			for _, existingPod := range podsToProcess {
				if err := pm.processExistingPod(existingPod, nodeInfo, pod); err != nil {
					errCh.SendErrorWithCancel(err, cancel)
					return
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
