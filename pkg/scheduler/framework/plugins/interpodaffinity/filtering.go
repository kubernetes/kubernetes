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
	"sync"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	"k8s.io/kubernetes/pkg/scheduler/nodeinfo"
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

	return &copy
}

// updateWithPod updates the preFilterState counters with the (anti)affinity matches for the given pod.
func (s *preFilterState) updateWithPod(updatedPod, pod *v1.Pod, node *v1.Node, multiplier int64) error {
	if s == nil {
		return nil
	}

	// Update matching existing anti-affinity terms.
	updatedPodAffinity := updatedPod.Spec.Affinity
	if updatedPodAffinity != nil && updatedPodAffinity.PodAntiAffinity != nil {
		antiAffinityTerms, err := getAffinityTerms(pod, schedutil.GetPodAntiAffinityTerms(updatedPodAffinity.PodAntiAffinity))
		if err != nil {
			return fmt.Errorf("error in getting anti-affinity terms of Pod %v: %v", updatedPod.Name, err)
		}
		s.topologyToMatchedExistingAntiAffinityTerms.updateWithAntiAffinityTerms(pod, node, antiAffinityTerms, multiplier)
	}

	// Update matching incoming pod (anti)affinity terms.
	affinity := pod.Spec.Affinity
	podNodeName := updatedPod.Spec.NodeName
	if affinity != nil && len(podNodeName) > 0 {
		if affinity.PodAffinity != nil {
			affinityTerms, err := getAffinityTerms(pod, schedutil.GetPodAffinityTerms(affinity.PodAffinity))
			if err != nil {
				return fmt.Errorf("error in getting affinity terms of Pod %v: %v", pod.Name, err)
			}
			s.topologyToMatchedAffinityTerms.updateWithAffinityTerms(updatedPod, node, affinityTerms, multiplier)
		}
		if affinity.PodAntiAffinity != nil {
			antiAffinityTerms, err := getAffinityTerms(pod, schedutil.GetPodAntiAffinityTerms(affinity.PodAntiAffinity))
			if err != nil {
				klog.Errorf("error in getting anti-affinity terms of Pod %v: %v", pod.Name, err)
			}
			s.topologyToMatchedAntiAffinityTerms.updateWithAntiAffinityTerms(updatedPod, node, antiAffinityTerms, multiplier)
		}
	}
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
func (m topologyToMatchedTermCount) updateWithAffinityTerms(targetPod *v1.Pod, targetPodNode *v1.Node, affinityTerms []*affinityTerm, value int64) {
	if podMatchesAllAffinityTerms(targetPod, affinityTerms) {
		for _, t := range affinityTerms {
			if topologyValue, ok := targetPodNode.Labels[t.topologyKey]; ok {
				pair := topologyPair{key: t.topologyKey, value: topologyValue}
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

// updateAntiAffinityTerms updates the topologyToMatchedTermCount map with the specified value
// for each anti-affinity term matched the target pod.
func (m topologyToMatchedTermCount) updateWithAntiAffinityTerms(targetPod *v1.Pod, targetPodNode *v1.Node, antiAffinityTerms []*affinityTerm, value int64) {
	// Check anti-affinity terms.
	for _, a := range antiAffinityTerms {
		if schedutil.PodMatchesTermsNamespaceAndSelector(targetPod, a.namespaces, a.selector) {
			if topologyValue, ok := targetPodNode.Labels[a.topologyKey]; ok {
				pair := topologyPair{key: a.topologyKey, value: topologyValue}
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

// A processed version of v1.PodAffinityTerm.
type affinityTerm struct {
	namespaces  sets.String
	selector    labels.Selector
	topologyKey string
}

// getAffinityTerms receives a Pod and affinity terms and returns the namespaces and
// selectors of the terms.
func getAffinityTerms(pod *v1.Pod, v1Terms []v1.PodAffinityTerm) ([]*affinityTerm, error) {
	if v1Terms == nil {
		return nil, nil
	}

	var terms []*affinityTerm
	for _, term := range v1Terms {
		namespaces := schedutil.GetNamespacesFromPodAffinityTerm(pod, &term)
		selector, err := metav1.LabelSelectorAsSelector(term.LabelSelector)
		if err != nil {
			return nil, err
		}
		terms = append(terms, &affinityTerm{namespaces: namespaces, selector: selector, topologyKey: term.TopologyKey})
	}
	return terms, nil
}

// podMatchesAllAffinityTerms returns true IFF the given pod matches all the given terms.
func podMatchesAllAffinityTerms(pod *v1.Pod, terms []*affinityTerm) bool {
	if len(terms) == 0 {
		return false
	}
	for _, term := range terms {
		if !schedutil.PodMatchesTermsNamespaceAndSelector(pod, term.namespaces, term.selector) {
			return false
		}
	}
	return true
}

// getTPMapMatchingExistingAntiAffinity calculates the following for each existing pod on each node:
// (1) Whether it has PodAntiAffinity
// (2) Whether any AffinityTerm matches the incoming pod
func getTPMapMatchingExistingAntiAffinity(pod *v1.Pod, allNodes []*nodeinfo.NodeInfo) (topologyToMatchedTermCount, error) {
	errCh := schedutil.NewErrorChannel()
	var lock sync.Mutex
	topologyMap := make(topologyToMatchedTermCount)

	appendResult := func(toAppend topologyToMatchedTermCount) {
		lock.Lock()
		defer lock.Unlock()
		topologyMap.append(toAppend)
	}

	ctx, cancel := context.WithCancel(context.Background())

	processNode := func(i int) {
		nodeInfo := allNodes[i]
		node := nodeInfo.Node()
		if node == nil {
			klog.Error("node not found")
			return
		}
		for _, existingPod := range nodeInfo.PodsWithAffinity() {
			existingPodTopologyMaps, err := getMatchingAntiAffinityTopologyPairsOfPod(pod, existingPod, node)
			if err != nil {
				errCh.SendErrorWithCancel(err, cancel)
				return
			}
			if existingPodTopologyMaps != nil {
				appendResult(existingPodTopologyMaps)
			}
		}
	}
	workqueue.ParallelizeUntil(ctx, 16, len(allNodes), processNode)

	if err := errCh.ReceiveError(); err != nil {
		return nil, err
	}

	return topologyMap, nil
}

// getTPMapMatchingIncomingAffinityAntiAffinity finds existing Pods that match affinity terms of the given "pod".
// It returns a topologyToMatchedTermCount that are checked later by the affinity
// predicate. With this topologyToMatchedTermCount available, the affinity predicate does not
// need to check all the pods in the cluster.
func getTPMapMatchingIncomingAffinityAntiAffinity(pod *v1.Pod, allNodes []*nodeinfo.NodeInfo) (topologyToMatchedTermCount, topologyToMatchedTermCount, error) {
	topologyPairsAffinityPodsMap := make(topologyToMatchedTermCount)
	topologyToMatchedExistingAntiAffinityTerms := make(topologyToMatchedTermCount)
	affinity := pod.Spec.Affinity
	if affinity == nil || (affinity.PodAffinity == nil && affinity.PodAntiAffinity == nil) {
		return topologyPairsAffinityPodsMap, topologyToMatchedExistingAntiAffinityTerms, nil
	}

	var lock sync.Mutex
	appendResult := func(nodeName string, nodeTopologyPairsAffinityPodsMap, nodeTopologyPairsAntiAffinityPodsMap topologyToMatchedTermCount) {
		lock.Lock()
		defer lock.Unlock()
		if len(nodeTopologyPairsAffinityPodsMap) > 0 {
			topologyPairsAffinityPodsMap.append(nodeTopologyPairsAffinityPodsMap)
		}
		if len(nodeTopologyPairsAntiAffinityPodsMap) > 0 {
			topologyToMatchedExistingAntiAffinityTerms.append(nodeTopologyPairsAntiAffinityPodsMap)
		}
	}

	affinityTerms, err := getAffinityTerms(pod, schedutil.GetPodAffinityTerms(affinity.PodAffinity))
	if err != nil {
		return nil, nil, err
	}

	antiAffinityTerms, err := getAffinityTerms(pod, schedutil.GetPodAntiAffinityTerms(affinity.PodAntiAffinity))
	if err != nil {
		return nil, nil, err
	}

	processNode := func(i int) {
		nodeInfo := allNodes[i]
		node := nodeInfo.Node()
		if node == nil {
			klog.Error("node not found")
			return
		}
		nodeTopologyPairsAffinityPodsMap := make(topologyToMatchedTermCount)
		nodeTopologyPairsAntiAffinityPodsMap := make(topologyToMatchedTermCount)
		for _, existingPod := range nodeInfo.Pods() {
			// Check affinity terms.
			nodeTopologyPairsAffinityPodsMap.updateWithAffinityTerms(existingPod, node, affinityTerms, 1)

			// Check anti-affinity terms.
			nodeTopologyPairsAntiAffinityPodsMap.updateWithAntiAffinityTerms(existingPod, node, antiAffinityTerms, 1)
		}

		if len(nodeTopologyPairsAffinityPodsMap) > 0 || len(nodeTopologyPairsAntiAffinityPodsMap) > 0 {
			appendResult(node.Name, nodeTopologyPairsAffinityPodsMap, nodeTopologyPairsAntiAffinityPodsMap)
		}
	}
	workqueue.ParallelizeUntil(context.Background(), 16, len(allNodes), processNode)

	return topologyPairsAffinityPodsMap, topologyToMatchedExistingAntiAffinityTerms, nil
}

// targetPodMatchesAffinityOfPod returns true if "targetPod" matches ALL affinity terms of
// "pod". This function does not check topology.
// So, whether the targetPod actually matches or not needs further checks for a specific
// node.
func targetPodMatchesAffinityOfPod(pod, targetPod *v1.Pod) bool {
	affinity := pod.Spec.Affinity
	if affinity == nil || affinity.PodAffinity == nil {
		return false
	}
	affinityTerms, err := getAffinityTerms(pod, schedutil.GetPodAffinityTerms(affinity.PodAffinity))
	if err != nil {
		klog.Errorf("error in getting affinity terms of Pod %v", pod.Name)
		return false
	}
	return podMatchesAllAffinityTerms(targetPod, affinityTerms)
}

// PreFilter invoked at the prefilter extension point.
func (pl *InterPodAffinity) PreFilter(ctx context.Context, cycleState *framework.CycleState, pod *v1.Pod) *framework.Status {
	var allNodes []*nodeinfo.NodeInfo
	var havePodsWithAffinityNodes []*nodeinfo.NodeInfo
	var err error
	if allNodes, err = pl.sharedLister.NodeInfos().List(); err != nil {
		return framework.NewStatus(framework.Error, fmt.Sprintf("failed to list NodeInfos: %v", err))
	}
	if havePodsWithAffinityNodes, err = pl.sharedLister.NodeInfos().HavePodsWithAffinityList(); err != nil {
		return framework.NewStatus(framework.Error, fmt.Sprintf("failed to list NodeInfos with pods with affinity: %v", err))
	}

	// existingPodAntiAffinityMap will be used later for efficient check on existing pods' anti-affinity
	existingPodAntiAffinityMap, err := getTPMapMatchingExistingAntiAffinity(pod, havePodsWithAffinityNodes)
	if err != nil {
		return framework.NewStatus(framework.Error, fmt.Sprintf("calculating preFilterState: %v", err))
	}
	// incomingPodAffinityMap will be used later for efficient check on incoming pod's affinity
	// incomingPodAntiAffinityMap will be used later for efficient check on incoming pod's anti-affinity
	incomingPodAffinityMap, incomingPodAntiAffinityMap, err := getTPMapMatchingIncomingAffinityAntiAffinity(pod, allNodes)
	if err != nil {
		return framework.NewStatus(framework.Error, fmt.Sprintf("calculating preFilterState: %v", err))
	}

	s := &preFilterState{
		topologyToMatchedAffinityTerms:             incomingPodAffinityMap,
		topologyToMatchedAntiAffinityTerms:         incomingPodAntiAffinityMap,
		topologyToMatchedExistingAntiAffinityTerms: existingPodAntiAffinityMap,
	}

	cycleState.Write(preFilterStateKey, s)
	return nil
}

// PreFilterExtensions returns prefilter extensions, pod add and remove.
func (pl *InterPodAffinity) PreFilterExtensions() framework.PreFilterExtensions {
	return pl
}

// AddPod from pre-computed data in cycleState.
func (pl *InterPodAffinity) AddPod(ctx context.Context, cycleState *framework.CycleState, podToSchedule *v1.Pod, podToAdd *v1.Pod, nodeInfo *nodeinfo.NodeInfo) *framework.Status {
	state, err := getPreFilterState(cycleState)
	if err != nil {
		return framework.NewStatus(framework.Error, err.Error())
	}
	state.updateWithPod(podToAdd, podToSchedule, nodeInfo.Node(), 1)
	return nil
}

// RemovePod from pre-computed data in cycleState.
func (pl *InterPodAffinity) RemovePod(ctx context.Context, cycleState *framework.CycleState, podToSchedule *v1.Pod, podToRemove *v1.Pod, nodeInfo *nodeinfo.NodeInfo) *framework.Status {
	state, err := getPreFilterState(cycleState)
	if err != nil {
		return framework.NewStatus(framework.Error, err.Error())
	}
	state.updateWithPod(podToRemove, podToSchedule, nodeInfo.Node(), -1)
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
func (pl *InterPodAffinity) satisfiesExistingPodsAntiAffinity(pod *v1.Pod, state *preFilterState, nodeInfo *nodeinfo.NodeInfo) (bool, error) {
	node := nodeInfo.Node()
	topologyMap := state.topologyToMatchedExistingAntiAffinityTerms

	// Iterate over topology pairs to get any of the pods being affected by
	// the scheduled pod anti-affinity terms
	for topologyKey, topologyValue := range node.Labels {
		if topologyMap[topologyPair{key: topologyKey, value: topologyValue}] > 0 {
			klog.V(10).Infof("Cannot schedule pod %+v onto node %v", pod.Name, node.Name)
			return false, nil
		}
	}
	return true, nil
}

//  nodeMatchesAllTopologyTerms checks whether "nodeInfo" matches topology of all the "terms" for the given "pod".
func nodeMatchesAllTopologyTerms(pod *v1.Pod, topologyPairs topologyToMatchedTermCount, nodeInfo *nodeinfo.NodeInfo, terms []v1.PodAffinityTerm) bool {
	node := nodeInfo.Node()
	for _, term := range terms {
		if topologyValue, ok := node.Labels[term.TopologyKey]; ok {
			pair := topologyPair{key: term.TopologyKey, value: topologyValue}
			if topologyPairs[pair] <= 0 {
				return false
			}
		} else {
			return false
		}
	}
	return true
}

//  nodeMatchesAnyTopologyTerm checks whether "nodeInfo" matches
//  topology of any "term" for the given "pod".
func nodeMatchesAnyTopologyTerm(pod *v1.Pod, topologyPairs topologyToMatchedTermCount, nodeInfo *nodeinfo.NodeInfo, terms []v1.PodAffinityTerm) bool {
	node := nodeInfo.Node()
	for _, term := range terms {
		if topologyValue, ok := node.Labels[term.TopologyKey]; ok {
			pair := topologyPair{key: term.TopologyKey, value: topologyValue}
			if topologyPairs[pair] > 0 {
				return true
			}
		}
	}
	return false
}

// getMatchingAntiAffinityTopologyPairs calculates the following for "existingPod" on given node:
// (1) Whether it has PodAntiAffinity
// (2) Whether ANY AffinityTerm matches the incoming pod
func getMatchingAntiAffinityTopologyPairsOfPod(newPod *v1.Pod, existingPod *v1.Pod, node *v1.Node) (topologyToMatchedTermCount, error) {
	affinity := existingPod.Spec.Affinity
	if affinity == nil || affinity.PodAntiAffinity == nil {
		return nil, nil
	}

	topologyMap := make(topologyToMatchedTermCount)
	for _, term := range schedutil.GetPodAntiAffinityTerms(affinity.PodAntiAffinity) {
		selector, err := metav1.LabelSelectorAsSelector(term.LabelSelector)
		if err != nil {
			return nil, err
		}
		namespaces := schedutil.GetNamespacesFromPodAffinityTerm(existingPod, &term)
		if schedutil.PodMatchesTermsNamespaceAndSelector(newPod, namespaces, selector) {
			if topologyValue, ok := node.Labels[term.TopologyKey]; ok {
				pair := topologyPair{key: term.TopologyKey, value: topologyValue}
				topologyMap[pair]++
			}
		}
	}
	return topologyMap, nil
}

// satisfiesPodsAffinityAntiAffinity checks if scheduling the pod onto this node would break any term of this pod.
// This function returns two boolean flags. The first boolean flag indicates whether the pod matches affinity rules
// or not. The second boolean flag indicates if the pod matches anti-affinity rules.
func (pl *InterPodAffinity) satisfiesPodsAffinityAntiAffinity(pod *v1.Pod,
	state *preFilterState, nodeInfo *nodeinfo.NodeInfo,
	affinity *v1.Affinity) (bool, bool, error) {
	node := nodeInfo.Node()
	if node == nil {
		return false, false, fmt.Errorf("node not found")
	}

	// Check all affinity terms.
	topologyToMatchedAffinityTerms := state.topologyToMatchedAffinityTerms
	if affinityTerms := schedutil.GetPodAffinityTerms(affinity.PodAffinity); len(affinityTerms) > 0 {
		matchExists := nodeMatchesAllTopologyTerms(pod, topologyToMatchedAffinityTerms, nodeInfo, affinityTerms)
		if !matchExists {
			// This pod may the first pod in a series that have affinity to themselves. In order
			// to not leave such pods in pending state forever, we check that if no other pod
			// in the cluster matches the namespace and selector of this pod and the pod matches
			// its own terms, then we allow the pod to pass the affinity check.
			if len(topologyToMatchedAffinityTerms) != 0 || !targetPodMatchesAffinityOfPod(pod, pod) {
				return false, false, nil
			}
		}
	}

	// Check all anti-affinity terms.
	topologyToMatchedAntiAffinityTerms := state.topologyToMatchedAntiAffinityTerms
	if antiAffinityTerms := schedutil.GetPodAntiAffinityTerms(affinity.PodAntiAffinity); len(antiAffinityTerms) > 0 {
		matchExists := nodeMatchesAnyTopologyTerm(pod, topologyToMatchedAntiAffinityTerms, nodeInfo, antiAffinityTerms)
		if matchExists {
			return true, false, nil
		}
	}

	return true, true, nil
}

// Filter invoked at the filter extension point.
// It checks if a pod can be scheduled on the specified node with pod affinity/anti-affinity configuration.
func (pl *InterPodAffinity) Filter(ctx context.Context, cycleState *framework.CycleState, pod *v1.Pod, nodeInfo *nodeinfo.NodeInfo) *framework.Status {
	state, err := getPreFilterState(cycleState)
	if err != nil {
		return framework.NewStatus(framework.Error, err.Error())
	}

	if s, err := pl.satisfiesExistingPodsAntiAffinity(pod, state, nodeInfo); !s || err != nil {
		if err != nil {
			return framework.NewStatus(framework.Error, err.Error())
		}
		return framework.NewStatus(framework.Unschedulable, ErrReasonAffinityNotMatch, ErrReasonExistingAntiAffinityRulesNotMatch)
	}

	// Now check if <pod> requirements will be satisfied on this node.
	affinity := pod.Spec.Affinity
	if affinity == nil || (affinity.PodAffinity == nil && affinity.PodAntiAffinity == nil) {
		return nil
	}
	if satisfiesAffinity, satisfiesAntiAffinity, err := pl.satisfiesPodsAffinityAntiAffinity(pod, state, nodeInfo, affinity); err != nil || !satisfiesAffinity || !satisfiesAntiAffinity {
		if err != nil {
			return framework.NewStatus(framework.Error, err.Error())
		}

		if !satisfiesAffinity {
			return framework.NewStatus(framework.UnschedulableAndUnresolvable, ErrReasonAffinityNotMatch, ErrReasonAffinityRulesNotMatch)
		}

		return framework.NewStatus(framework.Unschedulable, ErrReasonAffinityNotMatch, ErrReasonAntiAffinityRulesNotMatch)
	}

	return nil
}
