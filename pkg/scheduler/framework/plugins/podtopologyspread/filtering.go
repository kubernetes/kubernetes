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
	"maps"
	"math"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/component-helpers/scheduling/corev1/nodeaffinity"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

const preFilterStateKey = "PreFilter" + Name

// preFilterState computed at PreFilter and used at Filter.
// It combines CriticalPaths and TpValueToMatchNum to represent:
// (1) critical paths where the least pods are matched on each spread constraint.
// (2) number of pods matched on each spread constraint.
// A nil preFilterState denotes it's not set at all (in PreFilter phase);
// An empty preFilterState object denotes it's a legit state and is set in PreFilter phase.
// Fields are exported for comparison during testing.
type preFilterState struct {
	Constraints []topologySpreadConstraint
	// CriticalPaths is a slice indexed by constraint index.
	// Per each entry, we record 2 critical paths instead of all critical paths.
	// CriticalPaths[i][0].MatchNum always holds the minimum matching number.
	// CriticalPaths[i][1].MatchNum is always greater or equal to CriticalPaths[i][0].MatchNum, but
	// it's not guaranteed to be the 2nd minimum match number.
	CriticalPaths []*criticalPaths
	// TpValueToMatchNum is a slice indexed by constraint index.
	// Each entry is keyed with topology value, and valued with the number of matching pods.
	TpValueToMatchNum []map[string]int
}

// minMatchNum returns the global minimum for the calculation of skew while taking MinDomains into account.
func (s *preFilterState) minMatchNum(constraintID int, minDomains int32) (int, error) {
	paths := s.CriticalPaths[constraintID]

	minMatchNum := paths[0].MatchNum
	domainsNum := len(s.TpValueToMatchNum[constraintID])

	if domainsNum < int(minDomains) {
		// When the number of eligible domains with matching topology keys is less than `minDomains`,
		// it treats "global minimum" as 0.
		minMatchNum = 0
	}

	return minMatchNum, nil
}

// Clone makes a copy of the given state.
func (s *preFilterState) Clone() framework.StateData {
	if s == nil {
		return nil
	}
	copy := preFilterState{
		// Constraints are shared because they don't change.
		Constraints:       s.Constraints,
		CriticalPaths:     make([]*criticalPaths, len(s.CriticalPaths)),
		TpValueToMatchNum: make([]map[string]int, len(s.TpValueToMatchNum)),
	}
	for i, paths := range s.CriticalPaths {
		copy.CriticalPaths[i] = &criticalPaths{paths[0], paths[1]}
	}
	for i, tpMap := range s.TpValueToMatchNum {
		copy.TpValueToMatchNum[i] = maps.Clone(tpMap)
	}
	return &copy
}

// CAVEAT: the reason that `[2]criticalPath` can work is based on the implementation of current
// preemption algorithm, in particular the following 2 facts:
// Fact 1: we only preempt pods on the same node, instead of pods on multiple nodes.
// Fact 2: each node is evaluated on a separate copy of the preFilterState during its preemption cycle.
// If we plan to turn to a more complex algorithm like "arbitrary pods on multiple nodes", this
// structure needs to be revisited.
// Fields are exported for comparison during testing.
type criticalPaths [2]struct {
	// TopologyValue denotes the topology value mapping to topology key.
	TopologyValue string
	// MatchNum denotes the number of matching pods.
	MatchNum int
}

func newCriticalPaths() *criticalPaths {
	return &criticalPaths{{MatchNum: math.MaxInt32}, {MatchNum: math.MaxInt32}}
}

func (p *criticalPaths) update(tpVal string, num int) {
	// first verify if `tpVal` exists or not
	i := -1
	if tpVal == p[0].TopologyValue {
		i = 0
	} else if tpVal == p[1].TopologyValue {
		i = 1
	}

	if i >= 0 {
		// `tpVal` exists
		p[i].MatchNum = num
		if p[0].MatchNum > p[1].MatchNum {
			// swap paths[0] and paths[1]
			p[0], p[1] = p[1], p[0]
		}
	} else {
		// `tpVal` doesn't exist
		if num < p[0].MatchNum {
			// update paths[1] with paths[0]
			p[1] = p[0]
			// update paths[0]
			p[0].TopologyValue, p[0].MatchNum = tpVal, num
		} else if num < p[1].MatchNum {
			// update paths[1]
			p[1].TopologyValue, p[1].MatchNum = tpVal, num
		}
	}
}

// PreFilter invoked at the prefilter extension point.
func (pl *PodTopologySpread) PreFilter(ctx context.Context, cycleState *framework.CycleState, pod *v1.Pod, nodes []*framework.NodeInfo) (*framework.PreFilterResult, *framework.Status) {
	s, err := pl.calPreFilterState(ctx, pod, nodes)
	if err != nil {
		return nil, framework.AsStatus(err)
	} else if s != nil && len(s.Constraints) == 0 {
		return nil, framework.NewStatus(framework.Skip)
	}

	cycleState.Write(preFilterStateKey, s)
	return nil, nil
}

// PreFilterExtensions returns prefilter extensions, pod add and remove.
func (pl *PodTopologySpread) PreFilterExtensions() framework.PreFilterExtensions {
	return pl
}

// AddPod from pre-computed data in cycleState.
func (pl *PodTopologySpread) AddPod(ctx context.Context, cycleState *framework.CycleState, podToSchedule *v1.Pod, podInfoToAdd *framework.PodInfo, nodeInfo *framework.NodeInfo) *framework.Status {
	s, err := getPreFilterState(cycleState)
	if err != nil {
		return framework.AsStatus(err)
	}

	pl.updateWithPod(s, podInfoToAdd.Pod, podToSchedule, nodeInfo.Node(), 1)
	return nil
}

// RemovePod from pre-computed data in cycleState.
func (pl *PodTopologySpread) RemovePod(ctx context.Context, cycleState *framework.CycleState, podToSchedule *v1.Pod, podInfoToRemove *framework.PodInfo, nodeInfo *framework.NodeInfo) *framework.Status {
	s, err := getPreFilterState(cycleState)
	if err != nil {
		return framework.AsStatus(err)
	}

	pl.updateWithPod(s, podInfoToRemove.Pod, podToSchedule, nodeInfo.Node(), -1)
	return nil
}

func (pl *PodTopologySpread) updateWithPod(s *preFilterState, updatedPod, preemptorPod *v1.Pod, node *v1.Node, delta int) {
	if s == nil || updatedPod.Namespace != preemptorPod.Namespace || node == nil {
		return
	}
	if !nodeLabelsMatchSpreadConstraints(node.Labels, s.Constraints) {
		return
	}

	requiredSchedulingTerm := nodeaffinity.GetRequiredNodeAffinity(preemptorPod)
	if !pl.enableNodeInclusionPolicyInPodTopologySpread {
		// spreading is applied to nodes that pass those filters.
		// Ignore parsing errors for backwards compatibility.
		if match, _ := requiredSchedulingTerm.Match(node); !match {
			return
		}
	}

	podLabelSet := labels.Set(updatedPod.Labels)
	for i, constraint := range s.Constraints {
		if !constraint.Selector.Matches(podLabelSet) {
			continue
		}

		if pl.enableNodeInclusionPolicyInPodTopologySpread &&
			!constraint.matchNodeInclusionPolicies(preemptorPod, node, requiredSchedulingTerm) {
			continue
		}

		v := node.Labels[constraint.TopologyKey]
		s.TpValueToMatchNum[i][v] += delta
		s.CriticalPaths[i].update(v, s.TpValueToMatchNum[i][v])
	}
}

// getPreFilterState fetches a pre-computed preFilterState.
func getPreFilterState(cycleState *framework.CycleState) (*preFilterState, error) {
	c, err := cycleState.Read(preFilterStateKey)
	if err != nil {
		// preFilterState doesn't exist, likely PreFilter wasn't invoked.
		return nil, fmt.Errorf("reading %q from cycleState: %w", preFilterStateKey, err)
	}

	s, ok := c.(*preFilterState)
	if !ok {
		return nil, fmt.Errorf("%+v convert to podtopologyspread.preFilterState error", c)
	}
	return s, nil
}

type topologyCount struct {
	topologyValue string
	constraintID  int
	count         int
}

// calPreFilterState computes preFilterState describing how pods are spread on topologies.
func (pl *PodTopologySpread) calPreFilterState(ctx context.Context, pod *v1.Pod, allNodes []*framework.NodeInfo) (*preFilterState, error) {
	constraints, err := pl.getConstraints(pod)
	if err != nil {
		return nil, fmt.Errorf("get constraints from pod: %w", err)
	}
	if len(constraints) == 0 {
		return &preFilterState{}, nil
	}

	s := preFilterState{
		Constraints:       constraints,
		CriticalPaths:     make([]*criticalPaths, len(constraints)),
		TpValueToMatchNum: make([]map[string]int, len(constraints)),
	}
	for i := 0; i < len(constraints); i++ {
		s.TpValueToMatchNum[i] = make(map[string]int, sizeHeuristic(len(allNodes), constraints[i]))
	}

	tpCountsByNode := make([][]topologyCount, len(allNodes))
	requiredNodeAffinity := nodeaffinity.GetRequiredNodeAffinity(pod)
	processNode := func(n int) {
		nodeInfo := allNodes[n]
		node := nodeInfo.Node()

		if !pl.enableNodeInclusionPolicyInPodTopologySpread {
			// spreading is applied to nodes that pass those filters.
			// Ignore parsing errors for backwards compatibility.
			if match, _ := requiredNodeAffinity.Match(node); !match {
				return
			}
		}

		// Ensure current node's labels contains all topologyKeys in 'Constraints'.
		if !nodeLabelsMatchSpreadConstraints(node.Labels, constraints) {
			return
		}

		tpCounts := make([]topologyCount, 0, len(constraints))
		for i, c := range constraints {
			if pl.enableNodeInclusionPolicyInPodTopologySpread &&
				!c.matchNodeInclusionPolicies(pod, node, requiredNodeAffinity) {
				continue
			}

			value := node.Labels[c.TopologyKey]
			count := countPodsMatchSelector(nodeInfo.Pods, c.Selector, pod.Namespace)
			tpCounts = append(tpCounts, topologyCount{
				topologyValue: value,
				constraintID:  i,
				count:         count,
			})
		}
		tpCountsByNode[n] = tpCounts
	}
	pl.parallelizer.Until(ctx, len(allNodes), processNode, pl.Name())

	for _, tpCounts := range tpCountsByNode {
		// tpCounts might not hold all the constraints, so index can't be used here as constraintID.
		for _, tpCount := range tpCounts {
			s.TpValueToMatchNum[tpCount.constraintID][tpCount.topologyValue] += tpCount.count
		}
	}

	// calculate min match for each constraint and topology value
	for i := 0; i < len(constraints); i++ {
		s.CriticalPaths[i] = newCriticalPaths()

		for value, num := range s.TpValueToMatchNum[i] {
			s.CriticalPaths[i].update(value, num)
		}
	}

	return &s, nil
}

// Filter invoked at the filter extension point.
func (pl *PodTopologySpread) Filter(ctx context.Context, cycleState *framework.CycleState, pod *v1.Pod, nodeInfo *framework.NodeInfo) *framework.Status {
	node := nodeInfo.Node()

	s, err := getPreFilterState(cycleState)
	if err != nil {
		return framework.AsStatus(err)
	}

	// However, "empty" preFilterState is legit which tolerates every toSchedule Pod.
	if len(s.Constraints) == 0 {
		return nil
	}

	logger := klog.FromContext(ctx)
	podLabelSet := labels.Set(pod.Labels)
	for i, c := range s.Constraints {
		tpKey := c.TopologyKey
		tpVal, ok := node.Labels[tpKey]
		if !ok {
			logger.V(5).Info("Node doesn't have required topology label for spread constraint", "node", klog.KObj(node), "topologyKey", tpKey)
			return framework.NewStatus(framework.UnschedulableAndUnresolvable, ErrReasonNodeLabelNotMatch)
		}

		// judging criteria:
		// 'existing matching num' + 'if self-match (1 or 0)' - 'global minimum' <= 'maxSkew'
		minMatchNum, err := s.minMatchNum(i, c.MinDomains)
		if err != nil {
			logger.Error(err, "Internal error occurred while retrieving value precalculated in PreFilter", "topologyKey", tpKey, "paths", s.CriticalPaths[i])
			continue
		}

		selfMatchNum := 0
		if c.Selector.Matches(podLabelSet) {
			selfMatchNum = 1
		}

		matchNum := s.TpValueToMatchNum[i][tpVal]
		skew := matchNum + selfMatchNum - minMatchNum
		if skew > int(c.MaxSkew) {
			logger.V(5).Info("Node failed spreadConstraint: matchNum + selfMatchNum - minMatchNum > maxSkew", "node", klog.KObj(node), "topologyKey", tpKey, "matchNum", matchNum, "selfMatchNum", selfMatchNum, "minMatchNum", minMatchNum, "maxSkew", c.MaxSkew)
			return framework.NewStatus(framework.Unschedulable, ErrReasonConstraintsNotMatch)
		}
	}

	return nil
}

func sizeHeuristic(nodes int, constraint topologySpreadConstraint) int {
	if constraint.TopologyKey == v1.LabelHostname {
		return nodes
	}
	return 0
}
