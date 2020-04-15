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

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	"k8s.io/kubernetes/pkg/scheduler/internal/parallelize"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"
)

// preScoreStateKey is the key in CycleState to InterPodAffinity pre-computed data for Scoring.
const preScoreStateKey = "PreScore" + Name

type scoreMap map[string]map[string]int64

// preScoreState computed at PreScore and used at Score.
type preScoreState struct {
	topologyScore     scoreMap
	affinityTerms     []*weightedAffinityTerm
	antiAffinityTerms []*weightedAffinityTerm
}

// Clone implements the mandatory Clone interface. We don't really copy the data since
// there is no need for that.
func (s *preScoreState) Clone() framework.StateData {
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

func (m scoreMap) processTerm(
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
		if m[term.topologyKey] == nil {
			m[term.topologyKey] = make(map[string]int64)
		}
		m[term.topologyKey][tpValue] += int64(term.weight * int32(multiplier))
	}
	return
}

func (m scoreMap) processTerms(terms []*weightedAffinityTerm, podToCheck *v1.Pod, fixedNode *v1.Node, multiplier int) {
	for _, term := range terms {
		m.processTerm(term, podToCheck, fixedNode, multiplier)
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

func (pl *InterPodAffinity) processExistingPod(state *preScoreState, existingPod *v1.Pod, existingPodNodeInfo *framework.NodeInfo, incomingPod *v1.Pod, topoScore scoreMap) error {
	existingPodAffinity := existingPod.Spec.Affinity
	existingHasAffinityConstraints := existingPodAffinity != nil && existingPodAffinity.PodAffinity != nil
	existingHasAntiAffinityConstraints := existingPodAffinity != nil && existingPodAffinity.PodAntiAffinity != nil
	existingPodNode := existingPodNodeInfo.Node()

	// For every soft pod affinity term of <pod>, if <existingPod> matches the term,
	// increment <p.counts> for every node in the cluster with the same <term.TopologyKey>
	// value as that of <existingPods>`s node by the term`s weight.
	topoScore.processTerms(state.affinityTerms, existingPod, existingPodNode, 1)

	// For every soft pod anti-affinity term of <pod>, if <existingPod> matches the term,
	// decrement <p.counts> for every node in the cluster with the same <term.TopologyKey>
	// value as that of <existingPod>`s node by the term`s weight.
	topoScore.processTerms(state.antiAffinityTerms, existingPod, existingPodNode, -1)

	if existingHasAffinityConstraints {
		// For every hard pod affinity term of <existingPod>, if <pod> matches the term,
		// increment <p.counts> for every node in the cluster with the same <term.TopologyKey>
		// value as that of <existingPod>'s node by the constant <ipa.hardPodAffinityWeight>
		if pl.args.HardPodAffinityWeight > 0 {
			terms := existingPodAffinity.PodAffinity.RequiredDuringSchedulingIgnoredDuringExecution
			// TODO: Uncomment this block when implement RequiredDuringSchedulingRequiredDuringExecution.
			//if len(existingPodAffinity.PodAffinity.RequiredDuringSchedulingRequiredDuringExecution) != 0 {
			//	terms = append(terms, existingPodAffinity.PodAffinity.RequiredDuringSchedulingRequiredDuringExecution...)
			//}
			for i := range terms {
				term := &terms[i]
				processedTerm, err := newWeightedAffinityTerm(existingPod, term, pl.args.HardPodAffinityWeight)
				if err != nil {
					return err
				}
				topoScore.processTerm(processedTerm, incomingPod, existingPodNode, 1)
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

		topoScore.processTerms(terms, incomingPod, existingPodNode, 1)
	}
	if existingHasAntiAffinityConstraints {
		// For every soft pod anti-affinity term of <existingPod>, if <pod> matches the term,
		// decrement <pm.counts> for every node in the cluster with the same <term.TopologyKey>
		// value as that of <existingPod>'s node by the term's weight.
		terms, err := getWeightedAffinityTerms(existingPod, existingPodAffinity.PodAntiAffinity.PreferredDuringSchedulingIgnoredDuringExecution)
		if err != nil {
			return err
		}
		topoScore.processTerms(terms, incomingPod, existingPodNode, -1)
	}
	return nil
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

	state := &preScoreState{
		topologyScore:     make(map[string]map[string]int64),
		affinityTerms:     affinityTerms,
		antiAffinityTerms: antiAffinityTerms,
	}

	errCh := parallelize.NewErrorChannel()
	ctx, cancel := context.WithCancel(pCtx)
	processNode := func(i int) {
		nodeInfo := allNodes[i]
		if nodeInfo.Node() == nil {
			return
		}
		// Unless the pod being scheduled has affinity terms, we only
		// need to process pods with affinity in the node.
		podsToProcess := nodeInfo.PodsWithAffinity
		if hasAffinityConstraints || hasAntiAffinityConstraints {
			// We need to process all the pods.
			podsToProcess = nodeInfo.Pods
		}

		topoScore := make(scoreMap)
		for _, existingPod := range podsToProcess {
			if err := pl.processExistingPod(state, existingPod.Pod, nodeInfo, pod, topoScore); err != nil {
				errCh.SendErrorWithCancel(err, cancel)
				return
			}
		}
		if len(topoScore) > 0 {
			pl.Lock()
			state.topologyScore.append(topoScore)
			pl.Unlock()
		}
	}
	parallelize.Until(ctx, len(allNodes), processNode)
	if err := errCh.ReceiveError(); err != nil {
		return framework.NewStatus(framework.Error, err.Error())
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
// The "score" returned in this function is the matching number of pods on the `nodeName`,
// it is normalized later.
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
// The basic rule is: the bigger the score(matching number of pods) is, the smaller the
// final normalized score will be.
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
