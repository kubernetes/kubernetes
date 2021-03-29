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

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

const (
	// preFilterStateKey is the key in CycleState to InterPodAffinity pre-computed data for Filtering.
	// Using the name of the plugin will likely help us avoid collisions with other plugins.
	preFilterStateKey = "PreFilter" + Name

	// ErrReasonExistingAntiAffinityRulesNotMatch is used for ExistingPodsAntiAffinityRulesNotMatch predicate error.
	ErrReasonExistingAntiAffinityRulesNotMatch = "node(s) didn't satisfy existing pods anti-affinity rules"
	// ErrReasonAffinityNotMatch is used for MatchInterPodAffinity predicate error.
	ErrReasonAffinityNotMatch = "node(s) didn't match pod affinity/anti-affinity rules"
	// ErrReasonAffinityRulesNotMatch is used for PodAffinityRulesNotMatch predicate error.
	ErrReasonAffinityRulesNotMatch = "node(s) didn't match pod affinity rules"
	// ErrReasonAntiAffinityRulesNotMatch is used for PodAntiAffinityRulesNotMatch predicate error.
	ErrReasonAntiAffinityRulesNotMatch = "node(s) didn't match pod anti-affinity rules"
)

// preFilterState computed at PreFilter and used at Filter.
type preFilterState struct {
	// A map of topology pairs to the number of existing pods that has anti-affinity terms that match the "pod".
	existingAntiAffinityCounts topologyToMatchedTermCount
	// A map of topology pairs to the number of existing pods that match the affinity terms of the "pod".
	affinityCounts topologyToMatchedTermCount
	// A map of topology pairs to the number of existing pods that match the anti-affinity terms of the "pod".
	antiAffinityCounts topologyToMatchedTermCount
	// podInfo of the incoming pod.
	podInfo *framework.PodInfo
	// A copy of the incoming pod's namespace labels.
	namespaceLabels         labels.Set
	enableNamespaceSelector bool
}

// Clone the prefilter state.
func (s *preFilterState) Clone() framework.StateData {
	if s == nil {
		return nil
	}

	copy := preFilterState{}
	copy.affinityCounts = s.affinityCounts.clone()
	copy.antiAffinityCounts = s.antiAffinityCounts.clone()
	copy.existingAntiAffinityCounts = s.existingAntiAffinityCounts.clone()
	// No need to deep copy the podInfo because it shouldn't change.
	copy.podInfo = s.podInfo
	copy.namespaceLabels = s.namespaceLabels
	copy.enableNamespaceSelector = s.enableNamespaceSelector
	return &copy
}

// updateWithPod updates the preFilterState counters with the (anti)affinity matches for the given podInfo.
func (s *preFilterState) updateWithPod(pInfo *framework.PodInfo, node *v1.Node, multiplier int64) {
	if s == nil {
		return
	}

	s.existingAntiAffinityCounts.updateWithAntiAffinityTerms(pInfo.RequiredAntiAffinityTerms, s.podInfo.Pod, s.namespaceLabels, node, multiplier, s.enableNamespaceSelector)
	s.affinityCounts.updateWithAffinityTerms(s.podInfo.RequiredAffinityTerms, pInfo.Pod, node, multiplier, s.enableNamespaceSelector)
	// The incoming pod's terms have the namespaceSelector merged into the namespaces, and so
	// here we don't lookup the updated pod's namespace labels, hence passing nil for nsLabels.
	s.antiAffinityCounts.updateWithAntiAffinityTerms(s.podInfo.RequiredAntiAffinityTerms, pInfo.Pod, nil, node, multiplier, s.enableNamespaceSelector)
}

type topologyPair struct {
	key   string
	value string
}
type topologyToMatchedTermCount map[topologyPair]int64

func (m topologyToMatchedTermCount) append(toAppend topologyToMatchedTermCount) {
	for pair := range toAppend {
		m[pair] += toAppend[pair]
	}
}

func (m topologyToMatchedTermCount) clone() topologyToMatchedTermCount {
	copy := make(topologyToMatchedTermCount, len(m))
	copy.append(m)
	return copy
}

func (m topologyToMatchedTermCount) update(node *v1.Node, tk string, value int64) {
	if tv, ok := node.Labels[tk]; ok {
		pair := topologyPair{key: tk, value: tv}
		m[pair] += value
		// value could be negative, hence we delete the entry if it is down to zero.
		if m[pair] == 0 {
			delete(m, pair)
		}
	}
}

// updates the topologyToMatchedTermCount map with the specified value
// for each affinity term if "targetPod" matches ALL terms.
func (m topologyToMatchedTermCount) updateWithAffinityTerms(
	terms []framework.AffinityTerm, pod *v1.Pod, node *v1.Node, value int64, enableNamespaceSelector bool) {
	if podMatchesAllAffinityTerms(terms, pod, enableNamespaceSelector) {
		for _, t := range terms {
			m.update(node, t.TopologyKey, value)
		}
	}
}

// updates the topologyToMatchedTermCount map with the specified value
// for each anti-affinity term matched the target pod.
func (m topologyToMatchedTermCount) updateWithAntiAffinityTerms(terms []framework.AffinityTerm, pod *v1.Pod, nsLabels labels.Set, node *v1.Node, value int64, enableNamespaceSelector bool) {
	// Check anti-affinity terms.
	for _, t := range terms {
		if t.Matches(pod, nsLabels, enableNamespaceSelector) {
			m.update(node, t.TopologyKey, value)
		}
	}
}

// returns true IFF the given pod matches all the given terms.
func podMatchesAllAffinityTerms(terms []framework.AffinityTerm, pod *v1.Pod, enableNamespaceSelector bool) bool {
	if len(terms) == 0 {
		return false
	}
	for _, t := range terms {
		// The incoming pod NamespaceSelector was merged into the Namespaces set, and so
		// we are not explicitly passing in namespace labels.
		if !t.Matches(pod, nil, enableNamespaceSelector) {
			return false
		}
	}
	return true
}

// calculates the following for each existing pod on each node:
// (1) Whether it has PodAntiAffinity
// (2) Whether any AffinityTerm matches the incoming pod
func (pl *InterPodAffinity) getExistingAntiAffinityCounts(pod *v1.Pod, nsLabels labels.Set, nodes []*framework.NodeInfo, enableNamespaceSelector bool) topologyToMatchedTermCount {
	topoMaps := make([]topologyToMatchedTermCount, len(nodes))
	index := int32(-1)
	processNode := func(i int) {
		nodeInfo := nodes[i]
		node := nodeInfo.Node()
		if node == nil {
			klog.Error("node not found")
			return
		}
		topoMap := make(topologyToMatchedTermCount)
		for _, existingPod := range nodeInfo.PodsWithRequiredAntiAffinity {
			topoMap.updateWithAntiAffinityTerms(existingPod.RequiredAntiAffinityTerms, pod, nsLabels, node, 1, enableNamespaceSelector)
		}
		if len(topoMap) != 0 {
			topoMaps[atomic.AddInt32(&index, 1)] = topoMap
		}
	}
	pl.parallelizer.Until(context.Background(), len(nodes), processNode)

	result := make(topologyToMatchedTermCount)
	for i := 0; i <= int(index); i++ {
		result.append(topoMaps[i])
	}

	return result
}

// finds existing Pods that match affinity terms of the incoming pod's (anti)affinity terms.
// It returns a topologyToMatchedTermCount that are checked later by the affinity
// predicate. With this topologyToMatchedTermCount available, the affinity predicate does not
// need to check all the pods in the cluster.
func (pl *InterPodAffinity) getIncomingAffinityAntiAffinityCounts(podInfo *framework.PodInfo, allNodes []*framework.NodeInfo, enableNamespaceSelector bool) (topologyToMatchedTermCount, topologyToMatchedTermCount) {
	affinityCounts := make(topologyToMatchedTermCount)
	antiAffinityCounts := make(topologyToMatchedTermCount)
	if len(podInfo.RequiredAffinityTerms) == 0 && len(podInfo.RequiredAntiAffinityTerms) == 0 {
		return affinityCounts, antiAffinityCounts
	}

	affinityCountsList := make([]topologyToMatchedTermCount, len(allNodes))
	antiAffinityCountsList := make([]topologyToMatchedTermCount, len(allNodes))
	index := int32(-1)
	processNode := func(i int) {
		nodeInfo := allNodes[i]
		node := nodeInfo.Node()
		if node == nil {
			klog.Error("node not found")
			return
		}
		affinity := make(topologyToMatchedTermCount)
		antiAffinity := make(topologyToMatchedTermCount)
		for _, existingPod := range nodeInfo.Pods {
			affinity.updateWithAffinityTerms(podInfo.RequiredAffinityTerms, existingPod.Pod, node, 1, enableNamespaceSelector)
			// The incoming pod's terms have the namespaceSelector merged into the namespaces, and so
			// here we don't lookup the existing pod's namespace labels, hence passing nil for nsLabels.
			antiAffinity.updateWithAntiAffinityTerms(podInfo.RequiredAntiAffinityTerms, existingPod.Pod, nil, node, 1, enableNamespaceSelector)
		}

		if len(affinity) > 0 || len(antiAffinity) > 0 {
			k := atomic.AddInt32(&index, 1)
			affinityCountsList[k] = affinity
			antiAffinityCountsList[k] = antiAffinity
		}
	}
	pl.parallelizer.Until(context.Background(), len(allNodes), processNode)

	for i := 0; i <= int(index); i++ {
		affinityCounts.append(affinityCountsList[i])
		antiAffinityCounts.append(antiAffinityCountsList[i])
	}

	return affinityCounts, antiAffinityCounts
}

// PreFilter invoked at the prefilter extension point.
func (pl *InterPodAffinity) PreFilter(ctx context.Context, cycleState *framework.CycleState, pod *v1.Pod) *framework.Status {
	var allNodes []*framework.NodeInfo
	var nodesWithRequiredAntiAffinityPods []*framework.NodeInfo
	var err error
	if allNodes, err = pl.sharedLister.NodeInfos().List(); err != nil {
		return framework.AsStatus(fmt.Errorf("failed to list NodeInfos: %w", err))
	}
	if nodesWithRequiredAntiAffinityPods, err = pl.sharedLister.NodeInfos().HavePodsWithRequiredAntiAffinityList(); err != nil {
		return framework.AsStatus(fmt.Errorf("failed to list NodeInfos with pods with affinity: %w", err))
	}

	s := &preFilterState{
		enableNamespaceSelector: pl.enableNamespaceSelector,
	}

	s.podInfo = framework.NewPodInfo(pod)
	if s.podInfo.ParseError != nil {
		return framework.NewStatus(framework.UnschedulableAndUnresolvable, fmt.Sprintf("parsing pod: %+v", s.podInfo.ParseError))
	}

	if pl.enableNamespaceSelector {
		for i := range s.podInfo.RequiredAffinityTerms {
			if err := pl.mergeAffinityTermNamespacesIfNotEmpty(&s.podInfo.RequiredAffinityTerms[i]); err != nil {
				return framework.AsStatus(err)
			}
		}
		for i := range s.podInfo.RequiredAntiAffinityTerms {
			if err := pl.mergeAffinityTermNamespacesIfNotEmpty(&s.podInfo.RequiredAntiAffinityTerms[i]); err != nil {
				return framework.AsStatus(err)
			}
		}
		s.namespaceLabels = GetNamespaceLabelsSnapshot(pod.Namespace, pl.nsLister)
	}

	s.existingAntiAffinityCounts = pl.getExistingAntiAffinityCounts(pod, s.namespaceLabels, nodesWithRequiredAntiAffinityPods, pl.enableNamespaceSelector)
	s.affinityCounts, s.antiAffinityCounts = pl.getIncomingAffinityAntiAffinityCounts(s.podInfo, allNodes, pl.enableNamespaceSelector)

	cycleState.Write(preFilterStateKey, s)
	return nil
}

// PreFilterExtensions returns prefilter extensions, pod add and remove.
func (pl *InterPodAffinity) PreFilterExtensions() framework.PreFilterExtensions {
	return pl
}

// AddPod from pre-computed data in cycleState.
func (pl *InterPodAffinity) AddPod(ctx context.Context, cycleState *framework.CycleState, podToSchedule *v1.Pod, podInfoToAdd *framework.PodInfo, nodeInfo *framework.NodeInfo) *framework.Status {
	state, err := getPreFilterState(cycleState)
	if err != nil {
		return framework.AsStatus(err)
	}
	state.updateWithPod(podInfoToAdd, nodeInfo.Node(), 1)
	return nil
}

// RemovePod from pre-computed data in cycleState.
func (pl *InterPodAffinity) RemovePod(ctx context.Context, cycleState *framework.CycleState, podToSchedule *v1.Pod, podInfoToRemove *framework.PodInfo, nodeInfo *framework.NodeInfo) *framework.Status {
	state, err := getPreFilterState(cycleState)
	if err != nil {
		return framework.AsStatus(err)
	}
	state.updateWithPod(podInfoToRemove, nodeInfo.Node(), -1)
	return nil
}

func getPreFilterState(cycleState *framework.CycleState) (*preFilterState, error) {
	c, err := cycleState.Read(preFilterStateKey)
	if err != nil {
		// preFilterState doesn't exist, likely PreFilter wasn't invoked.
		return nil, fmt.Errorf("error reading %q from cycleState: %w", preFilterStateKey, err)
	}

	s, ok := c.(*preFilterState)
	if !ok {
		return nil, fmt.Errorf("%+v  convert to interpodaffinity.state error", c)
	}
	return s, nil
}

// Checks if scheduling the pod onto this node would break any anti-affinity
// terms indicated by the existing pods.
func satisfyExistingPodsAntiAffinity(state *preFilterState, nodeInfo *framework.NodeInfo) bool {
	if len(state.existingAntiAffinityCounts) > 0 {
		// Iterate over topology pairs to get any of the pods being affected by
		// the scheduled pod anti-affinity terms
		for topologyKey, topologyValue := range nodeInfo.Node().Labels {
			tp := topologyPair{key: topologyKey, value: topologyValue}
			if state.existingAntiAffinityCounts[tp] > 0 {
				return false
			}
		}
	}
	return true
}

//  Checks if the node satisfies the incoming pod's anti-affinity rules.
func satisfyPodAntiAffinity(state *preFilterState, nodeInfo *framework.NodeInfo) bool {
	if len(state.antiAffinityCounts) > 0 {
		for _, term := range state.podInfo.RequiredAntiAffinityTerms {
			if topologyValue, ok := nodeInfo.Node().Labels[term.TopologyKey]; ok {
				tp := topologyPair{key: term.TopologyKey, value: topologyValue}
				if state.antiAffinityCounts[tp] > 0 {
					return false
				}
			}
		}
	}
	return true
}

// Checks if the node satisfies the incoming pod's affinity rules.
func satisfyPodAffinity(state *preFilterState, nodeInfo *framework.NodeInfo) bool {
	podsExist := true
	for _, term := range state.podInfo.RequiredAffinityTerms {
		if topologyValue, ok := nodeInfo.Node().Labels[term.TopologyKey]; ok {
			tp := topologyPair{key: term.TopologyKey, value: topologyValue}
			if state.affinityCounts[tp] <= 0 {
				podsExist = false
			}
		} else {
			// All topology labels must exist on the node.
			return false
		}
	}

	if !podsExist {
		// This pod may be the first pod in a series that have affinity to themselves. In order
		// to not leave such pods in pending state forever, we check that if no other pod
		// in the cluster matches the namespace and selector of this pod, the pod matches
		// its own terms, and the node has all the requested topologies, then we allow the pod
		// to pass the affinity check.
		if len(state.affinityCounts) == 0 && podMatchesAllAffinityTerms(state.podInfo.RequiredAffinityTerms, state.podInfo.Pod, state.enableNamespaceSelector) {
			return true
		}
		return false
	}
	return true
}

// Filter invoked at the filter extension point.
// It checks if a pod can be scheduled on the specified node with pod affinity/anti-affinity configuration.
func (pl *InterPodAffinity) Filter(ctx context.Context, cycleState *framework.CycleState, pod *v1.Pod, nodeInfo *framework.NodeInfo) *framework.Status {
	if nodeInfo.Node() == nil {
		return framework.NewStatus(framework.Error, "node not found")
	}

	state, err := getPreFilterState(cycleState)
	if err != nil {
		return framework.AsStatus(err)
	}

	if !satisfyPodAffinity(state, nodeInfo) {
		return framework.NewStatus(framework.UnschedulableAndUnresolvable, ErrReasonAffinityNotMatch, ErrReasonAffinityRulesNotMatch)
	}

	if !satisfyPodAntiAffinity(state, nodeInfo) {
		return framework.NewStatus(framework.Unschedulable, ErrReasonAffinityNotMatch, ErrReasonAntiAffinityRulesNotMatch)
	}

	if !satisfyExistingPodsAntiAffinity(state, nodeInfo) {
		return framework.NewStatus(framework.Unschedulable, ErrReasonAffinityNotMatch, ErrReasonExistingAntiAffinityRulesNotMatch)
	}

	return nil
}
