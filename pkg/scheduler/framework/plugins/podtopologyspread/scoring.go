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

package podtopologyspread

import (
	"context"
	"fmt"
	"math"
	"sync/atomic"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog"
	pluginhelper "k8s.io/kubernetes/pkg/scheduler/framework/plugins/helper"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
)

const postFilterStateKey = "PostFilter" + Name

// postFilterState computed at PostFilter and used at Score.
type postFilterState struct {
	constraints []topologySpreadConstraint
	// nodeNameSet is a string set holding all node names which have all constraints[*].topologyKey present.
	nodeNameSet sets.String
	// topologyPairToPodCounts is keyed with topologyPair, and valued with the number of matching pods.
	topologyPairToPodCounts map[topologyPair]*int64
}

// Clone implements the mandatory Clone interface. We don't really copy the data since
// there is no need for that.
func (s *postFilterState) Clone() framework.StateData {
	return s
}

// initialize iterates "filteredNodes" to filter out the nodes which don't have required topologyKey(s),
// and initialize two maps:
// 1) s.topologyPairToPodCounts: keyed with both eligible topology pair and node names.
// 2) s.nodeNameSet: keyed with node name, and valued with a *int64 pointer for eligible node only.
func (s *postFilterState) initialize(pod *v1.Pod, filteredNodes []*v1.Node) error {
	constraints, err := filterTopologySpreadConstraints(pod.Spec.TopologySpreadConstraints, v1.ScheduleAnyway)
	if err != nil {
		return err
	}
	if constraints == nil {
		return nil
	}
	s.constraints = constraints
	for _, node := range filteredNodes {
		if !nodeLabelsMatchSpreadConstraints(node.Labels, s.constraints) {
			continue
		}
		for _, constraint := range s.constraints {
			pair := topologyPair{key: constraint.topologyKey, value: node.Labels[constraint.topologyKey]}
			if s.topologyPairToPodCounts[pair] == nil {
				s.topologyPairToPodCounts[pair] = new(int64)
			}
		}
		s.nodeNameSet.Insert(node.Name)
		// For those nodes which don't have all required topologyKeys present, it's intentional to leave
		// their entries absent in nodeNameSet, so that we're able to score them to 0 afterwards.
	}
	return nil
}

// PostFilter builds and writes cycle state used by Score and NormalizeScore.
func (pl *PodTopologySpread) PostFilter(
	ctx context.Context,
	cycleState *framework.CycleState,
	pod *v1.Pod,
	filteredNodes []*v1.Node,
	_ framework.NodeToStatusMap,
) *framework.Status {
	allNodes, err := pl.sharedLister.NodeInfos().List()
	if err != nil {
		return framework.NewStatus(framework.Error, fmt.Sprintf("error when getting all nodes: %v", err))
	}

	if len(filteredNodes) == 0 || len(allNodes) == 0 {
		// No nodes to score.
		return nil
	}

	state := &postFilterState{
		nodeNameSet:             sets.String{},
		topologyPairToPodCounts: make(map[topologyPair]*int64),
	}
	err = state.initialize(pod, filteredNodes)
	if err != nil {
		return framework.NewStatus(framework.Error, fmt.Sprintf("error when calculating postFilterState: %v", err))
	}

	// return if incoming pod doesn't have soft topology spread constraints.
	if state.constraints == nil {
		cycleState.Write(postFilterStateKey, state)
		return nil
	}

	processAllNode := func(i int) {
		nodeInfo := allNodes[i]
		node := nodeInfo.Node()
		if node == nil {
			return
		}
		// (1) `node` should satisfy incoming pod's NodeSelector/NodeAffinity
		// (2) All topologyKeys need to be present in `node`
		if !pluginhelper.PodMatchesNodeSelectorAndAffinityTerms(pod, node) ||
			!nodeLabelsMatchSpreadConstraints(node.Labels, state.constraints) {
			return
		}

		for _, c := range state.constraints {
			pair := topologyPair{key: c.topologyKey, value: node.Labels[c.topologyKey]}
			// If current topology pair is not associated with any candidate node,
			// continue to avoid unnecessary calculation.
			if state.topologyPairToPodCounts[pair] == nil {
				continue
			}

			// <matchSum> indicates how many pods (on current node) match the <constraint>.
			matchSum := int64(0)
			for _, existingPod := range nodeInfo.Pods() {
				if c.selector.Matches(labels.Set(existingPod.Labels)) {
					matchSum++
				}
			}
			atomic.AddInt64(state.topologyPairToPodCounts[pair], matchSum)
		}
	}
	workqueue.ParallelizeUntil(ctx, 16, len(allNodes), processAllNode)

	cycleState.Write(postFilterStateKey, state)
	return nil
}

// Score invoked at the Score extension point.
// The "score" returned in this function is the matching number of pods on the `nodeName`,
// it is normalized later.
func (pl *PodTopologySpread) Score(ctx context.Context, cycleState *framework.CycleState, pod *v1.Pod, nodeName string) (int64, *framework.Status) {
	nodeInfo, err := pl.sharedLister.NodeInfos().Get(nodeName)
	if err != nil || nodeInfo.Node() == nil {
		return 0, framework.NewStatus(framework.Error, fmt.Sprintf("getting node %q from Snapshot: %v, node is nil: %v", nodeName, err, nodeInfo.Node() == nil))
	}

	node := nodeInfo.Node()
	s, err := getPostFilterState(cycleState)
	if err != nil {
		return 0, framework.NewStatus(framework.Error, err.Error())
	}

	// Return if the node is not qualified.
	if _, ok := s.nodeNameSet[node.Name]; !ok {
		return 0, nil
	}

	// For each present <pair>, current node gets a credit of <matchSum>.
	// And we sum up <matchSum> and return it as this node's score.
	var score int64
	for _, c := range s.constraints {
		if tpVal, ok := node.Labels[c.topologyKey]; ok {
			pair := topologyPair{key: c.topologyKey, value: tpVal}
			matchSum := *s.topologyPairToPodCounts[pair]
			score += matchSum
		}
	}
	return score, nil
}

// NormalizeScore invoked after scoring all nodes.
func (pl *PodTopologySpread) NormalizeScore(ctx context.Context, cycleState *framework.CycleState, pod *v1.Pod, scores framework.NodeScoreList) *framework.Status {
	s, err := getPostFilterState(cycleState)
	if err != nil {
		return framework.NewStatus(framework.Error, err.Error())
	}
	if s == nil {
		return nil
	}

	// Calculate the summed <total> score and <minScore>.
	var minScore int64 = math.MaxInt64
	var total int64
	for _, score := range scores {
		// it's mandatory to check if <score.Name> is present in m.nodeNameSet
		if _, ok := s.nodeNameSet[score.Name]; !ok {
			continue
		}
		total += score.Score
		if score.Score < minScore {
			minScore = score.Score
		}
	}

	maxMinDiff := total - minScore
	for i := range scores {
		nodeInfo, err := pl.sharedLister.NodeInfos().Get(scores[i].Name)
		if err != nil {
			return framework.NewStatus(framework.Error, err.Error())
		}
		node := nodeInfo.Node()
		// Debugging purpose: print the score for each node.
		// Score must be a pointer here, otherwise it's always 0.
		if klog.V(10) {
			defer func(score *int64, nodeName string) {
				klog.Infof("%v -> %v: PodTopologySpread NormalizeScore, Score: (%d)", pod.Name, nodeName, *score)
			}(&scores[i].Score, node.Name)
		}

		if maxMinDiff == 0 {
			scores[i].Score = framework.MaxNodeScore
			continue
		}

		if _, ok := s.nodeNameSet[node.Name]; !ok {
			scores[i].Score = 0
			continue
		}

		flippedScore := total - scores[i].Score
		fScore := float64(framework.MaxNodeScore) * (float64(flippedScore) / float64(maxMinDiff))
		scores[i].Score = int64(fScore)
	}
	return nil
}

// ScoreExtensions of the Score plugin.
func (pl *PodTopologySpread) ScoreExtensions() framework.ScoreExtensions {
	return pl
}

func getPostFilterState(cycleState *framework.CycleState) (*postFilterState, error) {
	c, err := cycleState.Read(postFilterStateKey)
	if err != nil {
		return nil, fmt.Errorf("error reading %q from cycleState: %v", postFilterStateKey, err)
	}

	s, ok := c.(*postFilterState)
	if !ok {
		return nil, fmt.Errorf("%+v  convert to podtopologyspread.postFilterState error", c)
	}
	return s, nil
}
