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
	"k8s.io/klog/v2"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	"k8s.io/kubernetes/pkg/scheduler/internal/parallelize"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"
)

const (
	// preFilterStateKey is the key in CycleState to InterPodAffinity pre-computed data for Filtering.
	// Using the name of the plugin will likely help us avoid collisions with other plugins.
	preFilterStateKey = "PreFilter" + Name

	// ErrReasonExistingAntiAffinityRulesNotMatch is used for ExistingPodsAntiAffinityRulesNotMatch predicate error.
	ErrReasonExistingAntiAffinityRulesNotMatch = "node(s) didn't satisfy existing pods anti-affinity rules"
	// ErrReasonAffinityNotMatch is used for MatchInterPodAffinity predicate error.
	ErrReasonAffinityNotMatch = "node(s) didn't match pod affinity/anti-affinity"
	// ErrReasonAffinityRulesNotMatch is used for PodAffinityRulesNotMatch predicate error.
	ErrReasonAffinityRulesNotMatch = "node(s) didn't match pod affinity rules"
	// ErrReasonAntiAffinityRulesNotMatch is used for PodAntiAffinityRulesNotMatch predicate error.
	ErrReasonAntiAffinityRulesNotMatch = "node(s) didn't match pod anti-affinity rules"
)

// preFilterState computed at PreFilter and used at Filter.
type preFilterState struct {
	// A map of topology pairs to the number of existing pods that has anti-affinity terms that match the "pod".
	topologyToMatchedExistingAntiAffinityTerms topologyToMatchedTermCount
	// A map of topology pairs to the number of existing pods that match the affinity terms of the "pod".
	topologyToMatchedAffinityTerms topologyToMatchedTermCount
	// A map of topology pairs to the number of existing pods that match the anti-affinity terms of the "pod".
	topologyToMatchedAntiAffinityTerms topologyToMatchedTermCount
	// podInfo of the incoming pod.
	podInfo *framework.PodInfo
}

// Clone the prefilter state.
func (s *preFilterState) Clone() framework.StateData {
	if s == nil {
		return nil
	}

	copy := preFilterState{}
	copy.topologyToMatchedAffinityTerms = s.topologyToMatchedAffinityTerms.clone()
	copy.topologyToMatchedAntiAffinityTerms = s.topologyToMatchedAntiAffinityTerms.clone()
	copy.topologyToMatchedExistingAntiAffinityTerms = s.topologyToMatchedExistingAntiAffinityTerms.clone()
	// No need to deep copy the podInfo because it shouldn't change.
	copy.podInfo = s.podInfo

	return &copy
}

// updateWithPod updates the preFilterState counters with the (anti)affinity matches for the given pod.
func (s *preFilterState) updateWithPod(updatedPod *v1.Pod, node *v1.Node, multiplier int64) error {
	if s == nil {
		return nil
	}

	// Update matching existing anti-affinity terms.
	// TODO(#91058): AddPod/RemovePod should pass a *framework.PodInfo type instead of *v1.Pod.
	updatedPodInfo := framework.NewPodInfo(updatedPod)
	s.topologyToMatchedExistingAntiAffinityTerms.updateWithAntiAffinityTerms(s.podInfo.Pod, node, updatedPodInfo.RequiredAntiAffinityTerms, multiplier)

	// Update matching incoming pod (anti)affinity terms.
	s.topologyToMatchedAffinityTerms.updateWithAffinityTerms(updatedPod, node, s.podInfo.RequiredAffinityTerms, multiplier)
	s.topologyToMatchedAntiAffinityTerms.updateWithAntiAffinityTerms(updatedPod, node, s.podInfo.RequiredAntiAffinityTerms, multiplier)

	return nil
}

// TODO(Huang-Wei): It might be possible to use "make(map[topologyPair]*int64)" so that
// we can do atomic additions instead of using a global mutext, however we need to consider
// how to init each topologyToMatchedTermCount.
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

// updateWithAffinityTerms updates the topologyToMatchedTermCount map with the specified value
// for each affinity term if "targetPod" matches ALL terms.
func (m topologyToMatchedTermCount) updateWithAffinityTerms(targetPod *v1.Pod, targetPodNode *v1.Node, affinityTerms []framework.AffinityTerm, value int64) {
	if podMatchesAllAffinityTerms(targetPod, affinityTerms) {
		for _, t := range affinityTerms {
			if topologyValue, ok := targetPodNode.Labels[t.TopologyKey]; ok {
				pair := topologyPair{key: t.TopologyKey, value: topologyValue}
				m[pair] += value
				// value could be a negative value, hence we delete the entry if
				// the entry is down to zero.
				if m[pair] == 0 {
					delete(m, pair)
				}
			}
		}
	}
}

// updateWithAntiAffinityTerms updates the topologyToMatchedTermCount map with the specified value
// for each anti-affinity term matched the target pod.
func (m topologyToMatchedTermCount) updateWithAntiAffinityTerms(targetPod *v1.Pod, targetPodNode *v1.Node, antiAffinityTerms []framework.AffinityTerm, value int64) {
	// Check anti-affinity terms.
	for _, a := range antiAffinityTerms {
		if schedutil.PodMatchesTermsNamespaceAndSelector(targetPod, a.Namespaces, a.Selector) {
			if topologyValue, ok := targetPodNode.Labels[a.TopologyKey]; ok {
				pair := topologyPair{key: a.TopologyKey, value: topologyValue}
				m[pair] += value
				// value could be a negative value, hence we delete the entry if
				// the entry is down to zero.
				if m[pair] == 0 {
					delete(m, pair)
				}
			}
		}
	}
}

// podMatchesAllAffinityTerms returns true IFF the given pod matches all the given terms.
func podMatchesAllAffinityTerms(pod *v1.Pod, terms []framework.AffinityTerm) bool {
	if len(terms) == 0 {
		return false
	}
	for _, term := range terms {
		if !schedutil.PodMatchesTermsNamespaceAndSelector(pod, term.Namespaces, term.Selector) {
			return false
		}
	}
	return true
}

// getTPMapMatchingExistingAntiAffinity calculates the following for each existing pod on each node:
// (1) Whether it has PodAntiAffinity
// (2) Whether any AffinityTerm matches the incoming pod
func getTPMapMatchingExistingAntiAffinity(pod *v1.Pod, allNodes []*framework.NodeInfo) topologyToMatchedTermCount {
	topoMaps := make([]topologyToMatchedTermCount, len(allNodes))
	index := int32(-1)
	processNode := func(i int) {
		nodeInfo := allNodes[i]
		node := nodeInfo.Node()
		if node == nil {
			klog.Error("node not found")
			return
		}
		topoMap := make(topologyToMatchedTermCount)
		for _, existingPod := range nodeInfo.PodsWithAffinity {
			topoMap.updateWithAntiAffinityTerms(pod, node, existingPod.RequiredAntiAffinityTerms, 1)
		}
		if len(topoMap) != 0 {
			topoMaps[atomic.AddInt32(&index, 1)] = topoMap
		}
	}
	parallelize.Until(context.Background(), len(allNodes), processNode)

	result := make(topologyToMatchedTermCount)
	for i := 0; i <= int(index); i++ {
		result.append(topoMaps[i])
	}

	return result
}

// getTPMapMatchingIncomingAffinityAntiAffinity finds existing Pods that match affinity terms of the given "pod".
// It returns a topologyToMatchedTermCount that are checked later by the affinity
// predicate. With this topologyToMatchedTermCount available, the affinity predicate does not
// need to check all the pods in the cluster.
func getTPMapMatchingIncomingAffinityAntiAffinity(podInfo *framework.PodInfo, allNodes []*framework.NodeInfo) (topologyToMatchedTermCount, topologyToMatchedTermCount) {
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
			// Check affinity terms.
			affinity.updateWithAffinityTerms(existingPod.Pod, node, podInfo.RequiredAffinityTerms, 1)

			// Check anti-affinity terms.
			antiAffinity.updateWithAntiAffinityTerms(existingPod.Pod, node, podInfo.RequiredAntiAffinityTerms, 1)
		}

		if len(affinity) > 0 || len(antiAffinity) > 0 {
			k := atomic.AddInt32(&index, 1)
			affinityCountsList[k] = affinity
			antiAffinityCountsList[k] = antiAffinity
		}
	}
	parallelize.Until(context.Background(), len(allNodes), processNode)

	for i := 0; i <= int(index); i++ {
		affinityCounts.append(affinityCountsList[i])
		antiAffinityCounts.append(antiAffinityCountsList[i])
	}

	return affinityCounts, antiAffinityCounts
}

// PreFilter invoked at the prefilter extension point.
func (pl *InterPodAffinity) PreFilter(ctx context.Context, cycleState *framework.CycleState, pod *v1.Pod) *framework.Status {
	var allNodes []*framework.NodeInfo
	var havePodsWithAffinityNodes []*framework.NodeInfo
	var err error
	if allNodes, err = pl.sharedLister.NodeInfos().List(); err != nil {
		return framework.NewStatus(framework.Error, fmt.Sprintf("failed to list NodeInfos: %v", err))
	}
	if havePodsWithAffinityNodes, err = pl.sharedLister.NodeInfos().HavePodsWithAffinityList(); err != nil {
		return framework.NewStatus(framework.Error, fmt.Sprintf("failed to list NodeInfos with pods with affinity: %v", err))
	}

	podInfo := framework.NewPodInfo(pod)

	// existingPodAntiAffinityMap will be used later for efficient check on existing pods' anti-affinity
	existingPodAntiAffinityMap := getTPMapMatchingExistingAntiAffinity(pod, havePodsWithAffinityNodes)

	// incomingPodAffinityMap will be used later for efficient check on incoming pod's affinity
	// incomingPodAntiAffinityMap will be used later for efficient check on incoming pod's anti-affinity
	incomingPodAffinityMap, incomingPodAntiAffinityMap := getTPMapMatchingIncomingAffinityAntiAffinity(podInfo, allNodes)

	s := &preFilterState{
		topologyToMatchedAffinityTerms:             incomingPodAffinityMap,
		topologyToMatchedAntiAffinityTerms:         incomingPodAntiAffinityMap,
		topologyToMatchedExistingAntiAffinityTerms: existingPodAntiAffinityMap,
		podInfo: podInfo,
	}

	cycleState.Write(preFilterStateKey, s)
	return nil
}

// PreFilterExtensions returns prefilter extensions, pod add and remove.
func (pl *InterPodAffinity) PreFilterExtensions() framework.PreFilterExtensions {
	return pl
}

// AddPod from pre-computed data in cycleState.
func (pl *InterPodAffinity) AddPod(ctx context.Context, cycleState *framework.CycleState, podToSchedule *v1.Pod, podToAdd *v1.Pod, nodeInfo *framework.NodeInfo) *framework.Status {
	state, err := getPreFilterState(cycleState)
	if err != nil {
		return framework.NewStatus(framework.Error, err.Error())
	}
	state.updateWithPod(podToAdd, nodeInfo.Node(), 1)
	return nil
}

// RemovePod from pre-computed data in cycleState.
func (pl *InterPodAffinity) RemovePod(ctx context.Context, cycleState *framework.CycleState, podToSchedule *v1.Pod, podToRemove *v1.Pod, nodeInfo *framework.NodeInfo) *framework.Status {
	state, err := getPreFilterState(cycleState)
	if err != nil {
		return framework.NewStatus(framework.Error, err.Error())
	}
	state.updateWithPod(podToRemove, nodeInfo.Node(), -1)
	return nil
}

func getPreFilterState(cycleState *framework.CycleState) (*preFilterState, error) {
	c, err := cycleState.Read(preFilterStateKey)
	if err != nil {
		// preFilterState doesn't exist, likely PreFilter wasn't invoked.
		return nil, fmt.Errorf("error reading %q from cycleState: %v", preFilterStateKey, err)
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
	if len(state.topologyToMatchedExistingAntiAffinityTerms) > 0 {
		// Iterate over topology pairs to get any of the pods being affected by
		// the scheduled pod anti-affinity terms
		for topologyKey, topologyValue := range nodeInfo.Node().Labels {
			tp := topologyPair{key: topologyKey, value: topologyValue}
			if state.topologyToMatchedExistingAntiAffinityTerms[tp] > 0 {
				return false
			}
		}
	}
	return true
}

//  Checks if the node satisifies the incoming pod's anti-affinity rules.
func satisfyPodAntiAffinity(state *preFilterState, nodeInfo *framework.NodeInfo) bool {
	for _, term := range state.podInfo.RequiredAntiAffinityTerms {
		if topologyValue, ok := nodeInfo.Node().Labels[term.TopologyKey]; ok {
			tp := topologyPair{key: term.TopologyKey, value: topologyValue}
			if state.topologyToMatchedAntiAffinityTerms[tp] > 0 {
				return false
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
			if state.topologyToMatchedAffinityTerms[tp] <= 0 {
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
		podInfo := state.podInfo
		if len(state.topologyToMatchedAffinityTerms) == 0 && podMatchesAllAffinityTerms(podInfo.Pod, podInfo.RequiredAffinityTerms) {
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
		return framework.NewStatus(framework.Error, err.Error())
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
