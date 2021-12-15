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
	"sync"
	"sync/atomic"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/component-helpers/scheduling/corev1/nodeaffinity"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

const preFilterStateKey = "PreFilter" + Name
const nodeMatchCountStateKey = "NodeMatchCount" + Name

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

	podLabelSet := labels.Set(updatedPod.Labels)
	for _, constraint := range s.Constraints {
		if !constraint.Selector.Matches(podLabelSet) {
			continue
		}

		k, v := constraint.TopologyKey, node.Labels[constraint.TopologyKey]
		if k == v1.LabelHostname {
			continue
		}
		pair := topologyPair{key: k, value: v}
		*s.TpPairToMatchNum[pair] += delta

		s.TpKeyToCriticalPaths[k].update(v, *s.TpPairToMatchNum[pair])
	}
}

// nodeMatchCountState is used to implement an optimization which defers min match count calculation
// of the node topology constraint until Filter. It contains the following:
//
// (1) constraints: Topology spread constraint for this particular scheduling cycle, this is generated
// during PreFilter and is the same as the one from preFilterState.
// (2) globalMinCounted: A boolean to indicate if nodeCriticalPaths has been updated after scanning all
// nodes. Before scanning all nodes, nodeCriticalPaths may have already been updated by PreFilterExtensions.
// (3) nodeCriticalPaths: Similar to TpKeyToCriticalPaths inside preFilterState, but here the topology
// key is assumed to be hostname. This is used to track the global min match count, which works under
// the context of current preemption algorithm.
// (4) nodeMatchNum: Similar to TpPairToMatchNum inside preFilterState, but here the topology key is assumed
// to be hostname. This is used to track the pod match count per node.
//
// Note that the PreFilterExtensions could also update this state. When accessing from Filter, since it's
// called in parallel, it is required to acquire the lock.
type nodeMatchCountState struct {
	lock        sync.RWMutex
	constraints []topologySpreadConstraint
	// globalMinCounted indicates whether nodeCriticalPaths has been updated after scanning all nodes.
	globalMinCounted bool
	// nodeCriticalPaths is the critical paths with node hostname as its topology key.
	nodeCriticalPaths *criticalPaths
	// nodeMatchNum is a map of which its key is the hostname, and value is the number of matching pods.
	nodeMatchNum map[string]int32
}

// Clone makes a copy of the given state.
func (s *nodeMatchCountState) Clone() framework.StateData {
	if s == nil {
		return nil
	}
	c := nodeMatchCountState{
		constraints:       s.constraints,
		globalMinCounted:  s.globalMinCounted,
		nodeCriticalPaths: &criticalPaths{s.nodeCriticalPaths[0], s.nodeCriticalPaths[1]},
		nodeMatchNum:      make(map[string]int32, len(s.nodeMatchNum)),
	}
	for hostname, matchNum := range s.nodeMatchNum {
		c.nodeMatchNum[hostname] = matchNum
	}
	return &c
}

// updateWithPod is equivalent to preFilterState's updateWithPod, which is called from PreFilterExtensions.
func (s *nodeMatchCountState) updateWithPod(updatedPod, preemptorPod *v1.Pod, nodeInfo *framework.NodeInfo, delta int32) {
	node := nodeInfo.Node()
	if s == nil || updatedPod.Namespace != preemptorPod.Namespace || node == nil {
		return
	}
	if !nodeLabelsMatchSpreadConstraints(node.Labels, s.constraints) {
		return
	}

	podLabelSet := labels.Set(updatedPod.Labels)
	for _, constraint := range s.constraints {
		if !constraint.Selector.Matches(podLabelSet) {
			continue
		}

		k, v := constraint.TopologyKey, node.Labels[constraint.TopologyKey]
		if k != v1.LabelHostname {
			continue
		}

		if _, ok := s.nodeMatchNum[v]; !ok {
			s.nodeMatchNum[v] = int32(countPodsMatchSelector(nodeInfo.Pods, constraint.Selector, updatedPod.Namespace))
		} else {
			s.nodeMatchNum[v] += delta
		}

		s.nodeCriticalPaths.update(v, s.nodeMatchNum[v])
	}
}

// PreFilter invoked at the prefilter extension point.
func (pl *PodTopologySpread) PreFilter(ctx context.Context, cycleState *framework.CycleState, pod *v1.Pod) *framework.Status {
	s, err := pl.calPreFilterState(pod)
	if err != nil {
		return framework.AsStatus(err)
	}
	cycleState.Write(preFilterStateKey, s)

	// Initializes nodeMatchCountState only when there is node hostname topology constraint.
	for _, constraint := range s.Constraints {
		if constraint.TopologyKey == v1.LabelHostname {
			n := &nodeMatchCountState{
				constraints:       s.Constraints,
				nodeCriticalPaths: newCriticalPaths(),
				nodeMatchNum:      make(map[string]int32),
			}
			cycleState.Write(nodeMatchCountStateKey, n)
			break
		}
	}
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

	ns, err := getNodeMatchCountState(cycleState)
	if err == nil {
		// Ignore error since the state may not exist if the pod does
		// not contain node hostname topology constraint.
		ns.updateWithPod(podInfoToAdd.Pod, podToSchedule, nodeInfo, 1)
	}

	return nil
}

// RemovePod from pre-computed data in cycleState.
func (pl *PodTopologySpread) RemovePod(ctx context.Context, cycleState *framework.CycleState, podToSchedule *v1.Pod, podInfoToRemove *framework.PodInfo, nodeInfo *framework.NodeInfo) *framework.Status {
	s, err := getPreFilterState(cycleState)
	if err != nil {
		return framework.AsStatus(err)
	}

	s.updateWithPod(podInfoToRemove.Pod, podToSchedule, nodeInfo.Node(), -1)

	ns, err := getNodeMatchCountState(cycleState)
	if err == nil {
		// Ignore error since the state may not exist if the pod does
		// not contain node hostname topology constraint.
		ns.updateWithPod(podInfoToRemove.Pod, podToSchedule, nodeInfo, -1)
	}

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

// getNodeMatchCountState fetches nodeMatchCountState
func getNodeMatchCountState(cycleState *framework.CycleState) (*nodeMatchCountState, error) {
	c, err := cycleState.Read(nodeMatchCountStateKey)
	if err != nil {
		// nodeMatchCountStateKey doesn't exist, likely PreFilter wasn't invoked or
		// pod does not contain node hostname topology constraint.
		return nil, fmt.Errorf("reading %q from cycleState: %v", nodeMatchCountStateKey, err)
	}

	s, ok := c.(*nodeMatchCountState)
	if !ok {
		return nil, fmt.Errorf("%+v convert to podtopologyspread.nodeMatchCountState error", c)
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

	hostnameOnlyConstraint := true
	for _, c := range constraints {
		if c.TopologyKey != v1.LabelHostname {
			hostnameOnlyConstraint = false
			break
		}
	}
	if hostnameOnlyConstraint {
		return &s, nil
	}

	// Nodes that pass nodeAffinity check and carry all required topology keys will be
	// stored in `filteredNodes`, and be looped later to calculate preFilterState.
	var filteredNodes []*framework.NodeInfo
	requiredSchedulingTerm := nodeaffinity.GetRequiredNodeAffinity(pod)
	for _, n := range allNodes {
		node := n.Node()
		if shouldSkipNode(node, requiredSchedulingTerm, constraints) {
			continue
		}

		for _, c := range constraints {
			// per-node counts are calculated during Filter to optimize PreFilter
			// performance. Due to Filter only runs on a subset of Nodes,
			// calculating per-node counts is not necessary for all Nodes.
			if c.TopologyKey == v1.LabelHostname {
				continue
			}
			pair := topologyPair{key: c.TopologyKey, value: node.Labels[c.TopologyKey]}
			s.TpPairToMatchNum[pair] = new(int32)
		}

		filteredNodes = append(filteredNodes, n)
	}

	processNode := func(i int) {
		nodeInfo := filteredNodes[i]
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
	pl.parallelizer.Until(context.Background(), len(filteredNodes), processNode)

	// calculate min match for each topology pair
	for i := 0; i < len(constraints); i++ {
		key := constraints[i].TopologyKey
		if key == v1.LabelHostname {
			continue
		}
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

	allNodes, err := pl.sharedLister.NodeInfos().List()
	if err != nil {
		return framework.AsStatus(fmt.Errorf("failed to list NodeInfos: %v", err))
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

		var err error
		var skew, matchNum, minMatchNum int32
		if tpKey == v1.LabelHostname {
			matchNum = countNodeMatchCount(cycleState, tpVal, nodeInfo.Pods, c.Selector, pod.Namespace)
			skew = matchNum + selfMatchNum
			// Since global min match count is expected to be >= 0, and when skew is <= maxSkew, it is
			// not necessary to include global min match count in the calculation to conclude that
			// it will pass filter.
			if skew > c.MaxSkew {
				minMatchNum, err = pl.calNodeMinMatchCount(cycleState, allNodes, pod, c)
				if err != nil {
					// we log the error here, but do not do anything as if minMatchNum is 0
					// this may lead to pod failing to schedule when skew is in fact less than maxSkew
					// but this is ok in most cases
					klog.Errorf("internal error: fail to calculate per-node min match count for pod %s: %s", pod.Name, err)
				}
				skew -= minMatchNum
			}
		} else {
			pair := topologyPair{key: tpKey, value: tpVal}
			paths, ok := s.TpKeyToCriticalPaths[tpKey]
			if !ok {
				// error which should not happen
				klog.ErrorS(nil, "Internal error occurred while retrieving paths from topology key", "topologyKey", tpKey, "paths", s.TpKeyToCriticalPaths)
				continue
			}
			// judging criteria:
			// 'existing matching num' + 'if self-match (1 or 0)' - 'global min matching num' <= 'maxSkew'
			minMatchNum = paths[0].MatchNum

			if tpCount := s.TpPairToMatchNum[pair]; tpCount != nil {
				matchNum = *tpCount
			}
			skew = matchNum + selfMatchNum - minMatchNum
		}

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

// countNodeMatchCount returns the pod match count for a particular node if it's
// already cached in nodeMatchCountState, otherwise it calls countPodsMatchSelector
// to find the match count for the node and saves it to nodeMatchCountState.
func countNodeMatchCount(cycleState *framework.CycleState, hostname string, podInfos []*framework.PodInfo, selector labels.Selector, ns string) int32 {
	s, err := getNodeMatchCountState(cycleState)
	if err != nil {
		return int32(countPodsMatchSelector(podInfos, selector, ns))
	}

	s.lock.RLock()
	c, ok := s.nodeMatchNum[hostname]
	s.lock.RUnlock()
	if ok {
		return c
	}

	c = int32(countPodsMatchSelector(podInfos, selector, ns))
	s.lock.Lock()
	s.nodeMatchNum[hostname] = c
	s.lock.Unlock()
	return c
}

// calNodeMinMatchCount either returns the global min match count if it's already cached in
// nodeMatchCountState, or it computes the value by scanning all pods running on all nodes,
// and saves it to nodeMatchCountState.
func (pl *PodTopologySpread) calNodeMinMatchCount(cycleState *framework.CycleState, allNodes []*framework.NodeInfo, pod *v1.Pod, constraint topologySpreadConstraint) (int32, error) {
	s, err := getNodeMatchCountState(cycleState)
	if err != nil {
		return 0, fmt.Errorf("get nodeMatchCountState: %v", err)
	}

	s.lock.RLock()
	if s.globalMinCounted {
		minMatchCount := s.nodeCriticalPaths[0].MatchNum
		s.lock.RUnlock()
		return minMatchCount, nil
	}

	// Need to calculate global min, switch to acquire exclusive write lock
	s.lock.RUnlock()
	s.lock.Lock()
	defer s.lock.Unlock()

	// Check again in case another goroutine has already updated this
	if s.globalMinCounted {
		return s.nodeCriticalPaths[0].MatchNum, nil
	}

	// Allocate both remainingNodeMatchNum and remainingNodeInfos with estimated capacity
	remainingNodeMatchNum := make(map[string]*int32, len(allNodes)-len(s.nodeMatchNum))
	remainingNodeInfos := make([]*framework.NodeInfo, 0, len(allNodes)-len(s.nodeMatchNum))

	requiredSchedulingTerm := nodeaffinity.GetRequiredNodeAffinity(pod)
	for _, n := range allNodes {
		node := n.Node()
		if shouldSkipNode(node, requiredSchedulingTerm, s.constraints) {
			continue
		}

		tpVal := node.Labels[v1.LabelHostname]
		if _, ok := s.nodeMatchNum[tpVal]; ok {
			continue
		}

		remainingNodeMatchNum[tpVal] = new(int32)
		remainingNodeInfos = append(remainingNodeInfos, n)
	}

	processNode := func(i int) {
		nodeInfo := remainingNodeInfos[i]
		tpVal := nodeInfo.Node().Labels[v1.LabelHostname]
		tpCount := remainingNodeMatchNum[tpVal]
		count := countPodsMatchSelector(nodeInfo.Pods, constraint.Selector, pod.Namespace)
		atomic.AddInt32(tpCount, int32(count))
	}
	pl.parallelizer.Until(context.Background(), len(remainingNodeInfos), processNode)

	// Update nodeCriticalPaths with only remainingNodeMatchNum as it already contains
	// values from nodeMatchNum
	for tpVal, num := range remainingNodeMatchNum {
		s.nodeCriticalPaths.update(tpVal, *num)
	}

	s.globalMinCounted = true
	return s.nodeCriticalPaths[0].MatchNum, nil
}

// shouldSkipNode returns true if the node should be skipped when calculating pod topology
// constraint match count.
func shouldSkipNode(node *v1.Node, requiredSchedulingTerm nodeaffinity.RequiredNodeAffinity, constraints []topologySpreadConstraint) bool {
	if node == nil {
		klog.ErrorS(nil, "Node not found")
		return true
	}
	// In accordance to design, if NodeAffinity or NodeSelector is defined,
	// spreading is applied to nodes that pass those filters.
	// Ignore parsing errors for backwards compatibility.
	match, _ := requiredSchedulingTerm.Match(node)
	if !match {
		return true
	}
	// Ensure current node's labels contains all topologyKeys in 'Constraints'.
	if !nodeLabelsMatchSpreadConstraints(node.Labels, constraints) {
		return true
	}

	return false
}
