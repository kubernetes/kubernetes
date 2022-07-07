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
	"k8s.io/component-helpers/scheduling/corev1/nodeaffinity"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

const preFilterStateKey = "PreFilter" + Name

// preFilterState computed at PreFilter and used at Filter.
// It combines TpKeyToCriticalPaths and TpPairToMatchNum to represent:
// (1) critical paths where the least pods are matched on each spread constraint.
// (2) number of pods matched on each spread constraint.
// A nil preFilterState denotes it's not set at all (in PreFilter phase);
// An empty preFilterState object denotes it's a legit state and is set in PreFilter phase.
// Fields are exported for comparison during testing.
type preFilterState struct {
	Constraints []topologySpreadConstraint
	// We record 2 critical paths instead of all critical paths here.
	// criticalPaths[0].MatchNum always holds the minimum matching number.
	// criticalPaths[1].MatchNum is always greater or equal to criticalPaths[0].MatchNum, but
	// it's not guaranteed to be the 2nd minimum match number.
	TpKeyToCriticalPaths map[string]*criticalPaths
	// TpPairToMatchNum is keyed with topologyPair, and valued with the number of matching pods.
	TpPairToMatchNum map[topologyPair]*int32
}

// Clone makes a copy of the given state.
func (s *preFilterState) Clone() framework.StateData {
	if s == nil {
		return nil
	}
	copy := preFilterState{
		// Constraints are shared because they don't change.
		Constraints:          s.Constraints,
		TpKeyToCriticalPaths: make(map[string]*criticalPaths, len(s.TpKeyToCriticalPaths)),
		TpPairToMatchNum:     make(map[topologyPair]*int32, len(s.TpPairToMatchNum)),
	}
	for tpKey, paths := range s.TpKeyToCriticalPaths {
		copy.TpKeyToCriticalPaths[tpKey] = &criticalPaths{paths[0], paths[1]}
	}
	for tpPair, matchNum := range s.TpPairToMatchNum {
		copyPair := topologyPair{key: tpPair.key, value: tpPair.value}
		copyCount := *matchNum
		copy.TpPairToMatchNum[copyPair] = &copyCount
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
	MatchNum int32
}

func newCriticalPaths() *criticalPaths {
	return &criticalPaths{{MatchNum: math.MaxInt32}, {MatchNum: math.MaxInt32}}
}

func (p *criticalPaths) update(tpVal string, num int32) {
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

func (s *preFilterState) updateWithPod(updatedPod, preemptorPod *v1.Pod, node *v1.Node, delta int32) {
	if s == nil || updatedPod.Namespace != preemptorPod.Namespace || node == nil {
		return
	}
	if !nodeLabelsMatchSpreadConstraints(node.Labels, s.Constraints) {
		return
	}

	// We only need to update the counters for the nodes that match the topology spreading policy,
	// in other words, the node affinity.
	requiredSchedulingTerm := nodeaffinity.GetRequiredNodeAffinity(preemptorPod)
	if match, _ := requiredSchedulingTerm.Match(node); !match {
		return
	}

	podLabelSet := labels.Set(updatedPod.Labels)
	for _, constraint := range s.Constraints {
		if !constraint.Selector.Matches(podLabelSet) {
			continue
		}

		k, v := constraint.TopologyKey, node.Labels[constraint.TopologyKey]
		pair := topologyPair{key: k, value: v}
		*s.TpPairToMatchNum[pair] += delta

		s.TpKeyToCriticalPaths[k].update(v, *s.TpPairToMatchNum[pair])
	}
}

// PreFilter invoked at the prefilter extension point.
func (pl *PodTopologySpread) PreFilter(ctx context.Context, cycleState *framework.CycleState, pod *v1.Pod) *framework.Status {
	s, err := pl.calPreFilterState(pod)
	if err != nil {
		return framework.AsStatus(err)
	}
	cycleState.Write(preFilterStateKey, s)
	return nil
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

	s.updateWithPod(podInfoToAdd.Pod, podToSchedule, nodeInfo.Node(), 1)
	return nil
}

// RemovePod from pre-computed data in cycleState.
func (pl *PodTopologySpread) RemovePod(ctx context.Context, cycleState *framework.CycleState, podToSchedule *v1.Pod, podInfoToRemove *framework.PodInfo, nodeInfo *framework.NodeInfo) *framework.Status {
	s, err := getPreFilterState(cycleState)
	if err != nil {
		return framework.AsStatus(err)
	}

	s.updateWithPod(podInfoToRemove.Pod, podToSchedule, nodeInfo.Node(), -1)
	return nil
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

// calPreFilterState computes preFilterState describing how pods are spread on topologies.
func (pl *PodTopologySpread) calPreFilterState(pod *v1.Pod) (*preFilterState, error) {
	allNodes, err := pl.sharedLister.NodeInfos().List()
	if err != nil {
		return nil, fmt.Errorf("listing NodeInfos: %w", err)
	}
	var constraints []topologySpreadConstraint
	if len(pod.Spec.TopologySpreadConstraints) > 0 {
		// We have feature gating in APIServer to strip the spec
		// so don't need to re-check feature gate, just check length of Constraints.
		constraints, err = filterTopologySpreadConstraints(pod.Spec.TopologySpreadConstraints, v1.DoNotSchedule)
		if err != nil {
			return nil, fmt.Errorf("obtaining pod's hard topology spread constraints: %w", err)
		}
	} else {
		constraints, err = pl.buildDefaultConstraints(pod, v1.DoNotSchedule)
		if err != nil {
			return nil, fmt.Errorf("setting default hard topology spread constraints: %w", err)
		}
	}
	if len(constraints) == 0 {
		return &preFilterState{}, nil
	}

	s := preFilterState{
		Constraints:          constraints,
		TpKeyToCriticalPaths: make(map[string]*criticalPaths, len(constraints)),
		TpPairToMatchNum:     make(map[topologyPair]*int32, sizeHeuristic(len(allNodes), constraints)),
	}
	requiredSchedulingTerm := nodeaffinity.GetRequiredNodeAffinity(pod)
	for _, n := range allNodes {
		node := n.Node()
		if node == nil {
			klog.ErrorS(nil, "Node not found")
			continue
		}
		// In accordance to design, if NodeAffinity or NodeSelector is defined,
		// spreading is applied to nodes that pass those filters.
		// Ignore parsing errors for backwards compatibility.
		match, _ := requiredSchedulingTerm.Match(node)
		if !match {
			continue
		}
		// Ensure current node's labels contains all topologyKeys in 'Constraints'.
		if !nodeLabelsMatchSpreadConstraints(node.Labels, constraints) {
			continue
		}
		for _, c := range constraints {
			pair := topologyPair{key: c.TopologyKey, value: node.Labels[c.TopologyKey]}
			s.TpPairToMatchNum[pair] = new(int32)
		}
	}

	processNode := func(i int) {
		nodeInfo := allNodes[i]
		node := nodeInfo.Node()

		for _, constraint := range constraints {
			pair := topologyPair{key: constraint.TopologyKey, value: node.Labels[constraint.TopologyKey]}
			tpCount := s.TpPairToMatchNum[pair]
			if tpCount == nil {
				continue
			}
			count := countPodsMatchSelector(nodeInfo.Pods, constraint.Selector, pod.Namespace)
			atomic.AddInt32(tpCount, int32(count))
		}
	}
	pl.parallelizer.Until(context.Background(), len(allNodes), processNode)

	// calculate min match for each topology pair
	for i := 0; i < len(constraints); i++ {
		key := constraints[i].TopologyKey
		s.TpKeyToCriticalPaths[key] = newCriticalPaths()
	}
	for pair, num := range s.TpPairToMatchNum {
		s.TpKeyToCriticalPaths[pair.key].update(pair.value, *num)
	}

	return &s, nil
}

// Filter invoked at the filter extension point.
func (pl *PodTopologySpread) Filter(ctx context.Context, cycleState *framework.CycleState, pod *v1.Pod, nodeInfo *framework.NodeInfo) *framework.Status {
	node := nodeInfo.Node()
	if node == nil {
		return framework.AsStatus(fmt.Errorf("node not found"))
	}

	s, err := getPreFilterState(cycleState)
	if err != nil {
		return framework.AsStatus(err)
	}

	// However, "empty" preFilterState is legit which tolerates every toSchedule Pod.
	if len(s.Constraints) == 0 {
		return nil
	}

	podLabelSet := labels.Set(pod.Labels)
	for _, c := range s.Constraints {
		tpKey := c.TopologyKey
		tpVal, ok := node.Labels[c.TopologyKey]
		if !ok {
			klog.V(5).InfoS("Node doesn't have required label", "node", klog.KObj(node), "label", tpKey)
			return framework.NewStatus(framework.UnschedulableAndUnresolvable, ErrReasonNodeLabelNotMatch)
		}

		selfMatchNum := int32(0)
		if c.Selector.Matches(podLabelSet) {
			selfMatchNum = 1
		}

		pair := topologyPair{key: tpKey, value: tpVal}
		paths, ok := s.TpKeyToCriticalPaths[tpKey]
		if !ok {
			// error which should not happen
			klog.ErrorS(nil, "Internal error occurred while retrieving paths from topology key", "topologyKey", tpKey, "paths", s.TpKeyToCriticalPaths)
			continue
		}
		// judging criteria:
		// 'existing matching num' + 'if self-match (1 or 0)' - 'global min matching num' <= 'maxSkew'
		minMatchNum := paths[0].MatchNum
		matchNum := int32(0)
		if tpCount := s.TpPairToMatchNum[pair]; tpCount != nil {
			matchNum = *tpCount
		}
		skew := matchNum + selfMatchNum - minMatchNum
		if skew > c.MaxSkew {
			klog.V(5).InfoS("Node failed spreadConstraint: matchNum + selfMatchNum - minMatchNum > maxSkew", "node", klog.KObj(node), "topologyKey", tpKey, "matchNum", matchNum, "selfMatchNum", selfMatchNum, "minMatchNum", minMatchNum, "maxSkew", c.MaxSkew)
			return framework.NewStatus(framework.Unschedulable, ErrReasonConstraintsNotMatch)
		}
	}

	return nil
}

func sizeHeuristic(nodes int, constraints []topologySpreadConstraint) int {
	for _, c := range constraints {
		if c.TopologyKey == v1.LabelHostname {
			return nodes
		}
	}
	return 0
}
