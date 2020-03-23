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

const preScoreStateKey = "PreScore" + Name

// preScoreState computed at PreScore and used at Score.
// Fields are exported for comparison during testing.
type preScoreState struct {
	Constraints []topologySpreadConstraint
	// NodeNameSet is a string set holding all node names which have all Constraints[*].topologyKey present.
	NodeNameSet sets.String
	// TopologyPairToPodCounts is keyed with topologyPair, and valued with the number of matching pods.
	TopologyPairToPodCounts map[topologyPair]*int64
}

// Clone implements the mandatory Clone interface. We don't really copy the data since
// there is no need for that.
func (s *preScoreState) Clone() framework.StateData {
	return s
}

// initPreScoreState iterates "filteredNodes" to filter out the nodes which
// don't have required topologyKey(s), and initialize two maps:
// 1) s.TopologyPairToPodCounts: keyed with both eligible topology pair and node names.
// 2) s.NodeNameSet: keyed with node name, and valued with a *int64 pointer for eligible node only.
func (pl *PodTopologySpread) initPreScoreState(s *preScoreState, pod *v1.Pod, filteredNodes []*v1.Node) error {
	var err error
	if len(pod.Spec.TopologySpreadConstraints) > 0 {
		s.Constraints, err = filterTopologySpreadConstraints(pod.Spec.TopologySpreadConstraints, v1.ScheduleAnyway)
		if err != nil {
			return fmt.Errorf("obtaining pod's soft topology spread constraints: %v", err)
		}
	} else {
		s.Constraints, err = pl.defaultConstraints(pod, v1.ScheduleAnyway)
		if err != nil {
			return fmt.Errorf("setting default soft topology spread constraints: %v", err)
		}
	}
	if len(s.Constraints) == 0 {
		return nil
	}
	for _, node := range filteredNodes {
		if !nodeLabelsMatchSpreadConstraints(node.Labels, s.Constraints) {
			continue
		}
		for _, constraint := range s.Constraints {
			pair := topologyPair{key: constraint.TopologyKey, value: node.Labels[constraint.TopologyKey]}
			if s.TopologyPairToPodCounts[pair] == nil {
				s.TopologyPairToPodCounts[pair] = new(int64)
			}
		}
		s.NodeNameSet.Insert(node.Name)
		// For those nodes which don't have all required topologyKeys present, it's intentional to leave
		// their entries absent in NodeNameSet, so that we're able to score them to 0 afterwards.
	}
	return nil
}

// PreScore builds and writes cycle state used by Score and NormalizeScore.
func (pl *PodTopologySpread) PreScore(
	ctx context.Context,
	cycleState *framework.CycleState,
	pod *v1.Pod,
	filteredNodes []*v1.Node,
) *framework.Status {
	allNodes, err := pl.sharedLister.NodeInfos().List()
	if err != nil {
		return framework.NewStatus(framework.Error, fmt.Sprintf("error when getting all nodes: %v", err))
	}

	if len(filteredNodes) == 0 || len(allNodes) == 0 {
		// No nodes to score.
		return nil
	}

	state := &preScoreState{
		NodeNameSet:             sets.String{},
		TopologyPairToPodCounts: make(map[topologyPair]*int64),
	}
	err = pl.initPreScoreState(state, pod, filteredNodes)
	if err != nil {
		return framework.NewStatus(framework.Error, fmt.Sprintf("error when calculating preScoreState: %v", err))
	}

	// return if incoming pod doesn't have soft topology spread Constraints.
	if len(state.Constraints) == 0 {
		cycleState.Write(preScoreStateKey, state)
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
			!nodeLabelsMatchSpreadConstraints(node.Labels, state.Constraints) {
			return
		}

		for _, c := range state.Constraints {
			pair := topologyPair{key: c.TopologyKey, value: node.Labels[c.TopologyKey]}
			// If current topology pair is not associated with any candidate node,
			// continue to avoid unnecessary calculation.
			if state.TopologyPairToPodCounts[pair] == nil {
				continue
			}

			// <matchSum> indicates how many pods (on current node) match the <constraint>.
			matchSum := int64(0)
			for _, existingPod := range nodeInfo.Pods() {
				// Bypass terminating Pod (see #87621).
				if existingPod.DeletionTimestamp != nil || existingPod.Namespace != pod.Namespace {
					continue
				}
				if c.Selector.Matches(labels.Set(existingPod.Labels)) {
					matchSum++
				}
			}
			atomic.AddInt64(state.TopologyPairToPodCounts[pair], matchSum)
		}
	}
	workqueue.ParallelizeUntil(ctx, 16, len(allNodes), processAllNode)

	cycleState.Write(preScoreStateKey, state)
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
	s, err := getPreScoreState(cycleState)
	if err != nil {
		return 0, framework.NewStatus(framework.Error, err.Error())
	}

	// Return if the node is not qualified.
	if _, ok := s.NodeNameSet[node.Name]; !ok {
		return 0, nil
	}

	// For each present <pair>, current node gets a credit of <matchSum>.
	// And we sum up <matchSum> and return it as this node's score.
	var score int64
	for _, c := range s.Constraints {
		if tpVal, ok := node.Labels[c.TopologyKey]; ok {
			pair := topologyPair{key: c.TopologyKey, value: tpVal}
			matchSum := *s.TopologyPairToPodCounts[pair]
			score += matchSum
		}
	}
	return score, nil
}

// NormalizeScore invoked after scoring all nodes.
func (pl *PodTopologySpread) NormalizeScore(ctx context.Context, cycleState *framework.CycleState, pod *v1.Pod, scores framework.NodeScoreList) *framework.Status {
	s, err := getPreScoreState(cycleState)
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
		// it's mandatory to check if <score.Name> is present in m.NodeNameSet
		if _, ok := s.NodeNameSet[score.Name]; !ok {
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

		if _, ok := s.NodeNameSet[node.Name]; !ok {
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

func getPreScoreState(cycleState *framework.CycleState) (*preScoreState, error) {
	c, err := cycleState.Read(preScoreStateKey)
	if err != nil {
		return nil, fmt.Errorf("error reading %q from cycleState: %v", preScoreStateKey, err)
	}

	s, ok := c.(*preScoreState)
	if !ok {
		return nil, fmt.Errorf("%+v  convert to podtopologyspread.preScoreState error", c)
	}
	return s, nil
}
