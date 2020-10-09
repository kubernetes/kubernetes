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
	"sync/atomic"

	v1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/internal/parallelize"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"
)

// preScoreStateKey is the key in CycleState to InterPodAffinity pre-computed data for Scoring.
const preScoreStateKey = "PreScore" + Name

type scoreMap map[string]map[string]int64

// preScoreState computed at PreScore and used at Score.
type preScoreState struct {
	topologyScore scoreMap
	podInfo       *framework.PodInfo
}

// Clone implements the mandatory Clone interface. We don't really copy the data since
// there is no need for that.
func (s *preScoreState) Clone() framework.StateData {
	return s
}

func (m scoreMap) processTerm(
	term *framework.WeightedAffinityTerm,
	podToCheck *v1.Pod,
	fixedNode *v1.Node,
	multiplier int,
) {
	if len(fixedNode.Labels) == 0 {
		return
	}

	match := schedutil.PodMatchesTermsNamespaceAndSelector(podToCheck, term.Namespaces, term.Selector)
	tpValue, tpValueExist := fixedNode.Labels[term.TopologyKey]
	if match && tpValueExist {
		if m[term.TopologyKey] == nil {
			m[term.TopologyKey] = make(map[string]int64)
		}
		m[term.TopologyKey][tpValue] += int64(term.Weight * int32(multiplier))
	}
	return
}

func (m scoreMap) processTerms(terms []framework.WeightedAffinityTerm, podToCheck *v1.Pod, fixedNode *v1.Node, multiplier int) {
	for _, term := range terms {
		m.processTerm(&term, podToCheck, fixedNode, multiplier)
	}
}

func (m scoreMap) append(other scoreMap) {
	for topology, oScores := range other {
		scores := m[topology]
		if scores == nil {
			m[topology] = oScores
			continue
		}
		for k, v := range oScores {
			scores[k] += v
		}
	}
}

func (pl *InterPodAffinity) processExistingPod(
	state *preScoreState,
	existingPod *framework.PodInfo,
	existingPodNodeInfo *framework.NodeInfo,
	incomingPod *v1.Pod,
	topoScore scoreMap,
) {
	existingPodNode := existingPodNodeInfo.Node()

	// For every soft pod affinity term of <pod>, if <existingPod> matches the term,
	// increment <p.counts> for every node in the cluster with the same <term.TopologyKey>
	// value as that of <existingPods>`s node by the term`s weight.
	topoScore.processTerms(state.podInfo.PreferredAffinityTerms, existingPod.Pod, existingPodNode, 1)

	// For every soft pod anti-affinity term of <pod>, if <existingPod> matches the term,
	// decrement <p.counts> for every node in the cluster with the same <term.TopologyKey>
	// value as that of <existingPod>`s node by the term`s weight.
	topoScore.processTerms(state.podInfo.PreferredAntiAffinityTerms, existingPod.Pod, existingPodNode, -1)

	// For every hard pod affinity term of <existingPod>, if <pod> matches the term,
	// increment <p.counts> for every node in the cluster with the same <term.TopologyKey>
	// value as that of <existingPod>'s node by the constant <args.hardPodAffinityWeight>
	if pl.args.HardPodAffinityWeight > 0 {
		for _, term := range existingPod.RequiredAffinityTerms {
			t := framework.WeightedAffinityTerm{AffinityTerm: term, Weight: pl.args.HardPodAffinityWeight}
			topoScore.processTerm(&t, incomingPod, existingPodNode, 1)
		}
	}

	// For every soft pod affinity term of <existingPod>, if <pod> matches the term,
	// increment <p.counts> for every node in the cluster with the same <term.TopologyKey>
	// value as that of <existingPod>'s node by the term's weight.
	topoScore.processTerms(existingPod.PreferredAffinityTerms, incomingPod, existingPodNode, 1)

	// For every soft pod anti-affinity term of <existingPod>, if <pod> matches the term,
	// decrement <pm.counts> for every node in the cluster with the same <term.TopologyKey>
	// value as that of <existingPod>'s node by the term's weight.
	topoScore.processTerms(existingPod.PreferredAntiAffinityTerms, incomingPod, existingPodNode, -1)
}

// PreScore builds and writes cycle state used by Score and NormalizeScore.
func (pl *InterPodAffinity) PreScore(
	pCtx context.Context,
	cycleState *framework.CycleState,
	pod *v1.Pod,
	nodes []*v1.Node,
) *framework.Status {
	if len(nodes) == 0 {
		// No nodes to score.
		return nil
	}

	if pl.sharedLister == nil {
		return framework.NewStatus(framework.Error, fmt.Sprintf("InterPodAffinity PreScore with empty shared lister found"))
	}

	affinity := pod.Spec.Affinity
	hasPreferredAffinityConstraints := affinity != nil && affinity.PodAffinity != nil && len(affinity.PodAffinity.PreferredDuringSchedulingIgnoredDuringExecution) > 0
	hasPreferredAntiAffinityConstraints := affinity != nil && affinity.PodAntiAffinity != nil && len(affinity.PodAntiAffinity.PreferredDuringSchedulingIgnoredDuringExecution) > 0

	// Unless the pod being scheduled has preferred affinity terms, we only
	// need to process nodes hosting pods with affinity.
	var allNodes []*framework.NodeInfo
	var err error
	if hasPreferredAffinityConstraints || hasPreferredAntiAffinityConstraints {
		allNodes, err = pl.sharedLister.NodeInfos().List()
		if err != nil {
			framework.NewStatus(framework.Error, fmt.Sprintf("get all nodes from shared lister error, err: %v", err))
		}
	} else {
		allNodes, err = pl.sharedLister.NodeInfos().HavePodsWithAffinityList()
		if err != nil {
			framework.NewStatus(framework.Error, fmt.Sprintf("get pods with affinity list error, err: %v", err))
		}
	}

	podInfo := framework.NewPodInfo(pod)
	if podInfo.ParseError != nil {
		// Ideally we never reach here, because errors will be caught by PreFilter
		return framework.NewStatus(framework.Error, fmt.Sprintf("parsing pod: %+v", podInfo.ParseError))
	}

	state := &preScoreState{
		topologyScore: make(map[string]map[string]int64),
		podInfo:       podInfo,
	}

	topoScores := make([]scoreMap, len(allNodes))
	index := int32(-1)
	processNode := func(i int) {
		nodeInfo := allNodes[i]
		if nodeInfo.Node() == nil {
			return
		}
		// Unless the pod being scheduled has preferred affinity terms, we only
		// need to process pods with affinity in the node.
		podsToProcess := nodeInfo.PodsWithAffinity
		if hasPreferredAffinityConstraints || hasPreferredAntiAffinityConstraints {
			// We need to process all the pods.
			podsToProcess = nodeInfo.Pods
		}

		topoScore := make(scoreMap)
		for _, existingPod := range podsToProcess {
			pl.processExistingPod(state, existingPod, nodeInfo, pod, topoScore)
		}
		if len(topoScore) > 0 {
			topoScores[atomic.AddInt32(&index, 1)] = topoScore
		}
	}
	parallelize.Until(context.Background(), len(allNodes), processNode)

	for i := 0; i <= int(index); i++ {
		state.topologyScore.append(topoScores[i])
	}

	cycleState.Write(preScoreStateKey, state)
	return nil
}

func getPreScoreState(cycleState *framework.CycleState) (*preScoreState, error) {
	c, err := cycleState.Read(preScoreStateKey)
	if err != nil {
		return nil, fmt.Errorf("Error reading %q from cycleState: %v", preScoreStateKey, err)
	}

	s, ok := c.(*preScoreState)
	if !ok {
		return nil, fmt.Errorf("%+v  convert to interpodaffinity.preScoreState error", c)
	}
	return s, nil
}

// Score invoked at the Score extension point.
// The "score" returned in this function is the sum of weights got from cycleState which have its topologyKey matching with the node's labels.
// it is normalized later.
// Note: the returned "score" is positive for pod-affinity, and negative for pod-antiaffinity.
func (pl *InterPodAffinity) Score(ctx context.Context, cycleState *framework.CycleState, pod *v1.Pod, nodeName string) (int64, *framework.Status) {
	nodeInfo, err := pl.sharedLister.NodeInfos().Get(nodeName)
	if err != nil || nodeInfo.Node() == nil {
		return 0, framework.NewStatus(framework.Error, fmt.Sprintf("getting node %q from Snapshot: %v, node is nil: %v", nodeName, err, nodeInfo.Node() == nil))
	}
	node := nodeInfo.Node()

	s, err := getPreScoreState(cycleState)
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
func (pl *InterPodAffinity) NormalizeScore(ctx context.Context, cycleState *framework.CycleState, pod *v1.Pod, scores framework.NodeScoreList) *framework.Status {
	s, err := getPreScoreState(cycleState)
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
