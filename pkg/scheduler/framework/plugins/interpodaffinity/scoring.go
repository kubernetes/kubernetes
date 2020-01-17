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
	"context"
	"fmt"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	"k8s.io/kubernetes/pkg/scheduler/nodeinfo"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"
)

// postFilterStateKey is the key in CycleState to InterPodAffinity pre-computed data for Scoring.
const postFilterStateKey = "PostFilter" + Name

// postFilterState computed at PostFilter and used at Score.
type postFilterState struct {
	topologyScore     map[string]map[string]int64
	affinityTerms     []*weightedAffinityTerm
	antiAffinityTerms []*weightedAffinityTerm
}

// Clone implements the mandatory Clone interface. We don't really copy the data since
// there is no need for that.
func (s *postFilterState) Clone() framework.StateData {
	return s
}

// A "processed" representation of v1.WeightedAffinityTerm.
type weightedAffinityTerm struct {
	affinityTerm
	weight int32
}

func newWeightedAffinityTerm(pod *v1.Pod, term *v1.PodAffinityTerm, weight int32) (*weightedAffinityTerm, error) {
	namespaces := schedutil.GetNamespacesFromPodAffinityTerm(pod, term)
	selector, err := metav1.LabelSelectorAsSelector(term.LabelSelector)
	if err != nil {
		return nil, err
	}
	return &weightedAffinityTerm{affinityTerm: affinityTerm{namespaces: namespaces, selector: selector, topologyKey: term.TopologyKey}, weight: weight}, nil
}

func getWeightedAffinityTerms(pod *v1.Pod, v1Terms []v1.WeightedPodAffinityTerm) ([]*weightedAffinityTerm, error) {
	if v1Terms == nil {
		return nil, nil
	}

	var terms []*weightedAffinityTerm
	for i := range v1Terms {
		p, err := newWeightedAffinityTerm(pod, &v1Terms[i].PodAffinityTerm, v1Terms[i].Weight)
		if err != nil {
			return nil, err
		}
		terms = append(terms, p)
	}
	return terms, nil
}

func (pl *InterPodAffinity) processTerm(
	state *postFilterState,
	term *weightedAffinityTerm,
	podToCheck *v1.Pod,
	fixedNode *v1.Node,
	multiplier int,
) {
	if len(fixedNode.Labels) == 0 {
		return
	}

	match := schedutil.PodMatchesTermsNamespaceAndSelector(podToCheck, term.namespaces, term.selector)
	tpValue, tpValueExist := fixedNode.Labels[term.topologyKey]
	if match && tpValueExist {
		pl.Lock()
		if state.topologyScore[term.topologyKey] == nil {
			state.topologyScore[term.topologyKey] = make(map[string]int64)
		}
		state.topologyScore[term.topologyKey][tpValue] += int64(term.weight * int32(multiplier))
		pl.Unlock()
	}
	return
}

func (pl *InterPodAffinity) processTerms(state *postFilterState, terms []*weightedAffinityTerm, podToCheck *v1.Pod, fixedNode *v1.Node, multiplier int) error {
	for _, term := range terms {
		pl.processTerm(state, term, podToCheck, fixedNode, multiplier)
	}
	return nil
}

func (pl *InterPodAffinity) processExistingPod(state *postFilterState, existingPod *v1.Pod, existingPodNodeInfo *nodeinfo.NodeInfo, incomingPod *v1.Pod) error {
	existingPodAffinity := existingPod.Spec.Affinity
	existingHasAffinityConstraints := existingPodAffinity != nil && existingPodAffinity.PodAffinity != nil
	existingHasAntiAffinityConstraints := existingPodAffinity != nil && existingPodAffinity.PodAntiAffinity != nil
	existingPodNode := existingPodNodeInfo.Node()

	// For every soft pod affinity term of <pod>, if <existingPod> matches the term,
	// increment <p.counts> for every node in the cluster with the same <term.TopologyKey>
	// value as that of <existingPods>`s node by the term`s weight.
	pl.processTerms(state, state.affinityTerms, existingPod, existingPodNode, 1)

	// For every soft pod anti-affinity term of <pod>, if <existingPod> matches the term,
	// decrement <p.counts> for every node in the cluster with the same <term.TopologyKey>
	// value as that of <existingPod>`s node by the term`s weight.
	pl.processTerms(state, state.antiAffinityTerms, existingPod, existingPodNode, -1)

	if existingHasAffinityConstraints {
		// For every hard pod affinity term of <existingPod>, if <pod> matches the term,
		// increment <p.counts> for every node in the cluster with the same <term.TopologyKey>
		// value as that of <existingPod>'s node by the constant <ipa.hardPodAffinityWeight>
		if pl.hardPodAffinityWeight > 0 {
			terms := existingPodAffinity.PodAffinity.RequiredDuringSchedulingIgnoredDuringExecution
			// TODO: Uncomment this block when implement RequiredDuringSchedulingRequiredDuringExecution.
			//if len(existingPodAffinity.PodAffinity.RequiredDuringSchedulingRequiredDuringExecution) != 0 {
			//	terms = append(terms, existingPodAffinity.PodAffinity.RequiredDuringSchedulingRequiredDuringExecution...)
			//}
			for i := range terms {
				term := &terms[i]
				processedTerm, err := newWeightedAffinityTerm(existingPod, term, pl.hardPodAffinityWeight)
				if err != nil {
					return err
				}
				pl.processTerm(state, processedTerm, incomingPod, existingPodNode, 1)
			}
		}
		// For every soft pod affinity term of <existingPod>, if <pod> matches the term,
		// increment <p.counts> for every node in the cluster with the same <term.TopologyKey>
		// value as that of <existingPod>'s node by the term's weight.
		terms, err := getWeightedAffinityTerms(existingPod, existingPodAffinity.PodAffinity.PreferredDuringSchedulingIgnoredDuringExecution)
		if err != nil {
			klog.Error(err)
			return nil
		}

		pl.processTerms(state, terms, incomingPod, existingPodNode, 1)
	}
	if existingHasAntiAffinityConstraints {
		// For every soft pod anti-affinity term of <existingPod>, if <pod> matches the term,
		// decrement <pm.counts> for every node in the cluster with the same <term.TopologyKey>
		// value as that of <existingPod>'s node by the term's weight.
		terms, err := getWeightedAffinityTerms(existingPod, existingPodAffinity.PodAntiAffinity.PreferredDuringSchedulingIgnoredDuringExecution)
		if err != nil {
			return err
		}
		pl.processTerms(state, terms, incomingPod, existingPodNode, -1)
	}
	return nil
}

// PostFilter builds and writes cycle state used by Score and NormalizeScore.
func (pl *InterPodAffinity) PostFilter(
	pCtx context.Context,
	cycleState *framework.CycleState,
	pod *v1.Pod,
	nodes []*v1.Node,
	_ framework.NodeToStatusMap,
) *framework.Status {
	if len(nodes) == 0 {
		// No nodes to score.
		return nil
	}

	if pl.sharedLister == nil {
		return framework.NewStatus(framework.Error, fmt.Sprintf("BuildTopologyPairToScore with empty shared lister"))
	}

	affinity := pod.Spec.Affinity
	hasAffinityConstraints := affinity != nil && affinity.PodAffinity != nil
	hasAntiAffinityConstraints := affinity != nil && affinity.PodAntiAffinity != nil

	// Unless the pod being scheduled has affinity terms, we only
	// need to process nodes hosting pods with affinity.
	allNodes, err := pl.sharedLister.NodeInfos().HavePodsWithAffinityList()
	if err != nil {
		framework.NewStatus(framework.Error, fmt.Sprintf("get pods with affinity list error, err: %v", err))
	}
	if hasAffinityConstraints || hasAntiAffinityConstraints {
		allNodes, err = pl.sharedLister.NodeInfos().List()
		if err != nil {
			framework.NewStatus(framework.Error, fmt.Sprintf("get all nodes from shared lister error, err: %v", err))
		}
	}

	var affinityTerms []*weightedAffinityTerm
	var antiAffinityTerms []*weightedAffinityTerm
	if hasAffinityConstraints {
		if affinityTerms, err = getWeightedAffinityTerms(pod, affinity.PodAffinity.PreferredDuringSchedulingIgnoredDuringExecution); err != nil {
			klog.Error(err)
			return nil
		}
	}
	if hasAntiAffinityConstraints {
		if antiAffinityTerms, err = getWeightedAffinityTerms(pod, affinity.PodAntiAffinity.PreferredDuringSchedulingIgnoredDuringExecution); err != nil {
			klog.Error(err)
			return nil
		}
	}

	state := &postFilterState{
		topologyScore:     make(map[string]map[string]int64),
		affinityTerms:     affinityTerms,
		antiAffinityTerms: antiAffinityTerms,
	}

	errCh := schedutil.NewErrorChannel()
	ctx, cancel := context.WithCancel(pCtx)
	processNode := func(i int) {
		nodeInfo := allNodes[i]
		if nodeInfo.Node() == nil {
			return
		}
		// Unless the pod being scheduled has affinity terms, we only
		// need to process pods with affinity in the node.
		podsToProcess := nodeInfo.PodsWithAffinity()
		if hasAffinityConstraints || hasAntiAffinityConstraints {
			// We need to process all the pods.
			podsToProcess = nodeInfo.Pods()
		}

		for _, existingPod := range podsToProcess {
			if err := pl.processExistingPod(state, existingPod, nodeInfo, pod); err != nil {
				errCh.SendErrorWithCancel(err, cancel)
				return
			}
		}
	}
	workqueue.ParallelizeUntil(ctx, 16, len(allNodes), processNode)
	if err := errCh.ReceiveError(); err != nil {
		return framework.NewStatus(framework.Error, err.Error())
	}

	cycleState.Write(postFilterStateKey, state)
	return nil
}

func getPostFilterState(cycleState *framework.CycleState) (*postFilterState, error) {
	c, err := cycleState.Read(postFilterStateKey)
	if err != nil {
		return nil, fmt.Errorf("Error reading %q from cycleState: %v", postFilterStateKey, err)
	}

	s, ok := c.(*postFilterState)
	if !ok {
		return nil, fmt.Errorf("%+v  convert to interpodaffinity.postFilterState error", c)
	}
	return s, nil
}

// Score invoked at the Score extension point.
// The "score" returned in this function is the matching number of pods on the `nodeName`,
// it is normalized later.
func (pl *InterPodAffinity) Score(ctx context.Context, cycleState *framework.CycleState, pod *v1.Pod, nodeName string) (int64, *framework.Status) {
	nodeInfo, err := pl.sharedLister.NodeInfos().Get(nodeName)
	if err != nil || nodeInfo.Node() == nil {
		return 0, framework.NewStatus(framework.Error, fmt.Sprintf("getting node %q from Snapshot: %v, node is nil: %v", nodeName, err, nodeInfo.Node() == nil))
	}
	node := nodeInfo.Node()

	s, err := getPostFilterState(cycleState)
	if err != nil {
		return 0, framework.NewStatus(framework.Error, err.Error())
	}
	var score int64
	for tpKey, tpValues := range s.topologyScore {
		if v, exist := node.Labels[tpKey]; exist {
			score += tpValues[v]
		}
	}

	return score, nil
}

// NormalizeScore normalizes the score for each filteredNode.
// The basic rule is: the bigger the score(matching number of pods) is, the smaller the
// final normalized score will be.
func (pl *InterPodAffinity) NormalizeScore(ctx context.Context, cycleState *framework.CycleState, pod *v1.Pod, scores framework.NodeScoreList) *framework.Status {
	s, err := getPostFilterState(cycleState)
	if err != nil {
		return framework.NewStatus(framework.Error, err.Error())
	}
	if len(s.topologyScore) == 0 {
		return nil
	}

	var maxCount, minCount int64
	for i := range scores {
		score := scores[i].Score
		if score > maxCount {
			maxCount = score
		}
		if score < minCount {
			minCount = score
		}
	}

	maxMinDiff := maxCount - minCount
	for i := range scores {
		fScore := float64(0)
		if maxMinDiff > 0 {
			fScore = float64(framework.MaxNodeScore) * (float64(scores[i].Score-minCount) / float64(maxMinDiff))
		}

		scores[i].Score = int64(fScore)
	}

	return nil
}

// ScoreExtensions of the Score plugin.
func (pl *InterPodAffinity) ScoreExtensions() framework.ScoreExtensions {
	return pl
}
