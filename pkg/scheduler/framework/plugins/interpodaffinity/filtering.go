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
	"os"
	"sync"
	"sync/atomic"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/cacheplugin"
)

const (
	// preFilterStateKey is the key in CycleState to InterPodAffinity pre-computed data for Filtering.
	// Using the name of the plugin will likely help us avoid collisions with other plugins.
	preFilterStateKey = "PreFilter" + Name

	// ErrReasonExistingAntiAffinityRulesNotMatch is used for ExistingPodsAntiAffinityRulesNotMatch predicate error.
	ErrReasonExistingAntiAffinityRulesNotMatch = "node(s) didn't satisfy existing pods anti-affinity rules"
	// ErrReasonAffinityRulesNotMatch is used for PodAffinityRulesNotMatch predicate error.
	ErrReasonAffinityRulesNotMatch = "node(s) didn't match pod affinity rules"
	// ErrReasonAntiAffinityRulesNotMatch is used for PodAntiAffinityRulesNotMatch predicate error.
	ErrReasonAntiAffinityRulesNotMatch = "node(s) didn't match pod anti-affinity rules"
)

// preFilterState computed at PreFilter and used at Filter.
type preFilterState struct {
	updateLock sync.RWMutex

	// A map of topology pairs to the number of existing pods that has anti-affinity terms that match the "pod".
	existingAntiAffinityCounts topologyToMatchedTermCount
	// A map of topology pairs to the number of existing pods that match the affinity terms of the "pod".
	affinityCounts topologyToMatchedTermCount
	// A map of topology pairs to the number of existing pods that match the anti-affinity terms of the "pod".
	antiAffinityCounts topologyToMatchedTermCount
	// podInfo of the incoming pod.
	podInfo fwk.PodInfo
	// A copy of the incoming pod's namespace labels.
	namespaceLabels labels.Set
}

// Clone the prefilter state.
func (s *preFilterState) Clone() fwk.StateData {
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
	return &copy
}

// updateWithPod updates the preFilterState counters with the (anti)affinity matches for the given podInfo.
func (s *preFilterState) updateWithPod(pInfo fwk.PodInfo, node *v1.Node, multiplier int64) {
	if s == nil {
		return
	}
	s.updateLock.Lock()
	defer s.updateLock.Unlock()

	s.existingAntiAffinityCounts.updateWithAntiAffinityTerms(pInfo.GetRequiredAntiAffinityTerms(), s.podInfo.GetPod(), s.namespaceLabels, node, multiplier)
	s.affinityCounts.updateWithAffinityTerms(s.podInfo.GetRequiredAffinityTerms(), pInfo.GetPod(), node, multiplier)
	// The incoming pod's terms have the namespaceSelector merged into the namespaces, and so
	// here we don't lookup the updated pod's namespace labels, hence passing nil for nsLabels.
	s.antiAffinityCounts.updateWithAntiAffinityTerms(s.podInfo.GetRequiredAntiAffinityTerms(), pInfo.GetPod(), nil, node, multiplier)
}

type topologyPair struct {
	key   string
	value string
}
type topologyToMatchedTermCount map[topologyPair]int64

func (m topologyToMatchedTermCount) merge(toMerge topologyToMatchedTermCount) {
	for pair, count := range toMerge {
		m[pair] += count
	}
}

func (m topologyToMatchedTermCount) mergeWithList(toMerge topologyToMatchedTermCountList) {
	for _, tmtc := range toMerge {
		m[tmtc.topologyPair] += tmtc.count
	}
}

func (m topologyToMatchedTermCount) String() string {
	ret := "["
	for pair, count := range m {
		ret += fmt.Sprintf("(%s,%s):%v,", pair.key, pair.value, count)
	}
	ret += "]"
	return ret
}

func (m topologyToMatchedTermCount) clone() topologyToMatchedTermCount {
	copy := make(topologyToMatchedTermCount, len(m))
	copy.merge(m)
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
	terms []fwk.AffinityTerm, pod *v1.Pod, node *v1.Node, value int64) {
	if podMatchesAllAffinityTerms(terms, pod) {
		for _, t := range terms {
			m.update(node, t.TopologyKey, value)
		}
	}
}

// updates the topologyToMatchedTermCount map with the specified value
// for each anti-affinity term matched the target pod.
func (m topologyToMatchedTermCount) updateWithAntiAffinityTerms(terms []fwk.AffinityTerm, pod *v1.Pod, nsLabels labels.Set, node *v1.Node, value int64) {
	// Check anti-affinity terms.
	for _, t := range terms {
		if t.Matches(pod, nsLabels) {
			m.update(node, t.TopologyKey, value)
		}
	}
}

// topologyToMatchedTermCountList is a slice equivalent of topologyToMatchedTermCount map.
// The use of slice improves the performance of PreFilter,
// especially due to faster iteration when merging than with topologyToMatchedTermCount.
type topologyToMatchedTermCountList []topologyPairCount

type topologyPairCount struct {
	topologyPair topologyPair
	count        int64
}

func (m *topologyToMatchedTermCountList) append(node *v1.Node, tk string, value int64) (tmp topologyToMatchedTermCountList) {
	tmp = make(topologyToMatchedTermCountList, 0)
	if tv, ok := node.Labels[tk]; ok {
		pair := topologyPair{key: tk, value: tv}
		*m = append(*m, topologyPairCount{
			topologyPair: pair,
			count:        value,
		})
		tmp = append(tmp, topologyPairCount{
			topologyPair: pair,
			count:        value,
		})
	}
	return
}

// appends the specified value to the topologyToMatchedTermCountList
// for each affinity term if "targetPod" matches ALL terms.
func (m *topologyToMatchedTermCountList) appendWithAffinityTerms(
	terms []fwk.AffinityTerm, pod *v1.Pod, node *v1.Node, value int64) (tmp topologyToMatchedTermCountList) {
	tmp = make(topologyToMatchedTermCountList, 0)
	if podMatchesAllAffinityTerms(terms, pod) {
		for _, t := range terms {
			tmp = append(tmp, m.append(node, t.TopologyKey, value)...)
		}
	}
	return
}

// appends the specified value to the topologyToMatchedTermCountList
// for each anti-affinity term matched the target pod.
func (m *topologyToMatchedTermCountList) appendWithAntiAffinityTerms(terms []fwk.AffinityTerm, pod *v1.Pod, nsLabels labels.Set, node *v1.Node, value int64) (tmp topologyToMatchedTermCountList) {
	// Check anti-affinity terms.
	tmp = make(topologyToMatchedTermCountList, 0)
	for _, t := range terms {
		if t.Matches(pod, nsLabels) {
			tmp = append(tmp, m.append(node, t.TopologyKey, value)...)
		}
	}
	return
}

// returns true IFF the given pod matches all the given terms.
func podMatchesAllAffinityTerms(terms []fwk.AffinityTerm, pod *v1.Pod) bool {
	if len(terms) == 0 {
		return false
	}
	for _, t := range terms {
		// The incoming pod NamespaceSelector was merged into the Namespaces set, and so
		// we are not explicitly passing in namespace labels.
		if !t.Matches(pod, nil) {
			return false
		}
	}
	return true
}

// calculates the following for each existing pod on each node:
//  1. Whether it has PodAntiAffinity
//  2. Whether any AntiAffinityTerm matches the incoming pod
func (pl *InterPodAffinity) getExistingAntiAffinityCounts(ctx context.Context, pod *v1.Pod, nsLabels labels.Set, nodes []fwk.NodeInfo) (topologyToMatchedTermCount,
	map[cacheplugin.NamespaceedNameNode]topologyToMatchedTermCount) {
	topoMapsByPod := make([]map[cacheplugin.NamespaceedNameNode]topologyToMatchedTermCountList, len(nodes))
	antiAffinityCountsList := make([]topologyToMatchedTermCountList, len(nodes))
	antiAffinityCountsByPods := make([]map[cacheplugin.NamespaceedNameNode]topologyToMatchedTermCountList, len(nodes))
	index := int32(-1)
	processNode := func(i int) {
		nodeInfo := nodes[i]
		node := nodeInfo.Node()

		topoMapByPod := make(map[cacheplugin.NamespaceedNameNode]topologyToMatchedTermCountList)
		antiAffinityCounts := make(topologyToMatchedTermCountList, 0)
		for _, existingPod := range nodeInfo.GetPodsWithRequiredAntiAffinity() {
			tmp := antiAffinityCounts.appendWithAntiAffinityTerms(existingPod.GetRequiredAntiAffinityTerms(), pod, nsLabels, node, 1)
			if len(tmp) > 0 {
				topoMapByPod[cacheplugin.NamespaceedNameNode{Namespace: existingPod.GetPod().Namespace,
					Name: existingPod.GetPod().Name}] = tmp
			}
		}
		if len(antiAffinityCounts) != 0 {
			idx := atomic.AddInt32(&index, 1)
			antiAffinityCountsList[idx] = antiAffinityCounts
			antiAffinityCountsByPods[idx] = topoMapByPod
			topoMapsByPod[idx] = topoMapByPod
		}
	}
	pl.parallelizer.Until(ctx, len(nodes), processNode, pl.Name())

	result := make(topologyToMatchedTermCount)
	// Traditional for loop is slightly faster in this case than its "for range" equivalent.
	for i := 0; i <= int(index); i++ {
		result.mergeWithList(antiAffinityCountsList[i])
	}
	resultMpByPod := make(map[cacheplugin.NamespaceedNameNode]topologyToMatchedTermCount)
	for i := 0; i <= int(index); i++ {
		for k, v := range topoMapsByPod[i] {
			count := make(topologyToMatchedTermCount)
			count.mergeWithList(v)
			resultMpByPod[k] = count
		}
	}

	return result, resultMpByPod
}

// finds existing Pods that match affinity terms of the incoming pod's (anti)affinity terms.
// It returns a topologyToMatchedTermCount that are checked later by the affinity
// predicate. With this topologyToMatchedTermCount available, the affinity predicate does not
// need to check all the pods in the cluster.
func (pl *InterPodAffinity) getIncomingAffinityAntiAffinityCounts(ctx context.Context, podInfo fwk.PodInfo, allNodes []fwk.NodeInfo) (topologyToMatchedTermCount, topologyToMatchedTermCount,
	map[cacheplugin.NamespaceedNameNode][]topologyToMatchedTermCount) {
	affinityCounts := make(topologyToMatchedTermCount)
	antiAffinityCounts := make(topologyToMatchedTermCount)
	tpByPod := make(map[cacheplugin.NamespaceedNameNode][]topologyToMatchedTermCount)
	if len(podInfo.GetRequiredAffinityTerms()) == 0 && len(podInfo.GetRequiredAntiAffinityTerms()) == 0 {
		return affinityCounts, antiAffinityCounts, tpByPod
	}

	affinityCountsList := make([]topologyToMatchedTermCountList, len(allNodes))
	antiAffinityCountsList := make([]topologyToMatchedTermCountList, len(allNodes))
	topoMapsByPod := make([]map[cacheplugin.NamespaceedNameNode][]topologyToMatchedTermCountList, len(allNodes))
	index := int32(-1)
	processNode := func(i int) {
		nodeInfo := allNodes[i]
		node := nodeInfo.Node()

		affinity := make(topologyToMatchedTermCountList, 0)
		antiAffinity := make(topologyToMatchedTermCountList, 0)
		topoMapByPod := make(map[cacheplugin.NamespaceedNameNode][]topologyToMatchedTermCountList)
		for _, existingPod := range nodeInfo.GetPods() {
			af := affinity.appendWithAffinityTerms(podInfo.GetRequiredAffinityTerms(), existingPod.GetPod(), node, 1)
			// The incoming pod's terms have the namespaceSelector merged into the namespaces, and so
			// here we don't lookup the existing pod's namespace labels, hence passing nil for nsLabels.
			anaf := antiAffinity.appendWithAntiAffinityTerms(podInfo.GetRequiredAntiAffinityTerms(), existingPod.GetPod(), nil, node, 1)
			if len(af) > 0 || len(anaf) > 0 {
				topoMapByPod[cacheplugin.NamespaceedNameNode{Namespace: existingPod.GetPod().Namespace,
					Name: existingPod.GetPod().Name}] = []topologyToMatchedTermCountList{af, anaf}
			}
		}

		if len(affinity) > 0 || len(antiAffinity) > 0 {
			k := atomic.AddInt32(&index, 1)
			affinityCountsList[k] = affinity
			antiAffinityCountsList[k] = antiAffinity
			topoMapsByPod[k] = topoMapByPod
		}
	}
	pl.parallelizer.Until(ctx, len(allNodes), processNode, pl.Name())

	for i := 0; i <= int(index); i++ {
		affinityCounts.mergeWithList(affinityCountsList[i])
		antiAffinityCounts.mergeWithList(antiAffinityCountsList[i])
		for k, v := range topoMapsByPod[i] {
			count := []topologyToMatchedTermCount{topologyToMatchedTermCount{}, topologyToMatchedTermCount{}}
			count[0].mergeWithList(v[0])
			count[1].mergeWithList(v[1])
			tpByPod[k] = count
		}
	}

	return affinityCounts, antiAffinityCounts, tpByPod
}

// PreFilter invoked at the prefilter extension point.
func (pl *InterPodAffinity) PreFilter(ctx context.Context, cycleState fwk.CycleState, pod *v1.Pod, allNodes []fwk.NodeInfo) (*fwk.PreFilterResult, *fwk.Status) {
	var nodesWithRequiredAntiAffinityPods []fwk.NodeInfo
	var err error
	if nodesWithRequiredAntiAffinityPods, err = pl.sharedLister.NodeInfos().HavePodsWithRequiredAntiAffinityList(); err != nil {
		return nil, fwk.AsStatus(fmt.Errorf("failed to list NodeInfos with pods with affinity: %w", err))
	}

	s := &preFilterState{}

	if s.podInfo, err = framework.NewPodInfo(pod); err != nil {
		return nil, fwk.NewStatus(fwk.UnschedulableAndUnresolvable, fmt.Sprintf("parsing pod: %+v", err))
	}

	for i := range s.podInfo.GetRequiredAffinityTerms() {
		if err := pl.mergeAffinityTermNamespacesIfNotEmpty(s.podInfo.GetRequiredAffinityTerms()[i]); err != nil {
			return nil, fwk.AsStatus(err)
		}
	}
	for i := range s.podInfo.GetRequiredAntiAffinityTerms() {
		if err := pl.mergeAffinityTermNamespacesIfNotEmpty(s.podInfo.GetRequiredAntiAffinityTerms()[i]); err != nil {
			return nil, fwk.AsStatus(err)
		}
	}
	logger := klog.FromContext(ctx)
	// check existing required antiaffinity
	enableInterPodAffinityCache := os.Getenv("EnableInterPodAffinityCache") == "true"
	var existingCacheData *FilteringExistingPodAffinityTermDetailedState
	if enableInterPodAffinityCache {
		existingCacheData = pl.filteringExistingPodCache.impl.Read(namespacedLabels{namespace: pod.Namespace, labels: pod.Labels})
	}
	if enableInterPodAffinityCache && existingCacheData != nil {
		existingCacheData.lock.RLock()
		defer existingCacheData.lock.RUnlock()
		s.namespaceLabels = existingCacheData.namespaceLabels
		s.existingAntiAffinityCounts = existingCacheData.preCalRes.clone()
		logger.V(5).Info("found existingCacheData", "existingAntiAffinityCounts", s.existingAntiAffinityCounts.String(), "namespaceLabels", s.namespaceLabels)
	} else {
		s.namespaceLabels = GetNamespaceLabelsSnapshot(logger, pod.Namespace, pl.nsLister)
		if nodesWithRequiredAntiAffinityPods, err = pl.sharedLister.NodeInfos().HavePodsWithRequiredAntiAffinityList(); err != nil {
			return nil, fwk.AsStatus(fmt.Errorf("failed to list NodeInfos with pods with affinity: %w", err))
		}
		var mpByPod map[cacheplugin.NamespaceedNameNode]topologyToMatchedTermCount
		s.existingAntiAffinityCounts, mpByPod = pl.getExistingAntiAffinityCounts(ctx, pod, s.namespaceLabels, nodesWithRequiredAntiAffinityPods)
		if enableInterPodAffinityCache {
			logger.V(5).Info("set existingCacheData", "existingAntiAffinityCounts", s.existingAntiAffinityCounts.String(), "namespaceLabels", s.namespaceLabels)
			pl.filteringExistingPodCache.impl.Write(namespacedLabels{namespace: pod.Namespace, labels: pod.Labels}, NewFilteringExistingPodAffinityTermDetailedState(
				s.existingAntiAffinityCounts, mpByPod, pod.Namespace, pod.Labels, s.namespaceLabels))
		}
	}

	if len(s.existingAntiAffinityCounts) == 0 && len(s.podInfo.GetRequiredAffinityTerms()) == 0 && len(s.podInfo.GetRequiredAntiAffinityTerms()) == 0 {
		return nil, fwk.NewStatus(fwk.Skip)
	}
	tphash, hasTemplateHashInPod := pod.Annotations[apps.DefaultDeploymentUniqueLabelKey]
	var incomingCacheData *FilteringIncomingPodAffinityTermDetailedState
	if enableInterPodAffinityCache && hasTemplateHashInPod {
		incomingCacheData = pl.filteringIncomingPodCache.impl.Read(tphash)
	}
	if enableInterPodAffinityCache && incomingCacheData != nil {
		incomingCacheData.lock.RLock()
		defer incomingCacheData.lock.RUnlock()
		s.affinityCounts = incomingCacheData.affinityCounts.clone()
		s.antiAffinityCounts = incomingCacheData.antiAffinityCounts.clone()
		logger.V(5).Info("found incomingCacheData", "affinityCounts", s.affinityCounts.String(), "antiAffinityCounts", s.antiAffinityCounts.String())
	} else {
		// check incoming required antiaffinity
		if allNodes, err = pl.sharedLister.NodeInfos().List(); err != nil {
			return nil, fwk.AsStatus(fmt.Errorf("failed to list NodeInfos: %w", err))
		}
		for i := range s.podInfo.GetRequiredAffinityTerms() {
			if err := pl.mergeAffinityTermNamespacesIfNotEmpty(s.podInfo.GetRequiredAffinityTerms()[i]); err != nil {
				return nil, fwk.AsStatus(err)
			}
		}
		for i := range s.podInfo.GetRequiredAntiAffinityTerms() {
			if err := pl.mergeAffinityTermNamespacesIfNotEmpty(s.podInfo.GetRequiredAntiAffinityTerms()[i]); err != nil {
				return nil, fwk.AsStatus(err)
			}
		}
		var tpByPod map[cacheplugin.NamespaceedNameNode][]topologyToMatchedTermCount
		s.affinityCounts, s.antiAffinityCounts, tpByPod = pl.getIncomingAffinityAntiAffinityCounts(ctx, s.podInfo, allNodes)
		if enableInterPodAffinityCache {
			logger.V(5).Info("set incomingCacheData", "affinityCounts", s.affinityCounts.String(), "antiAffinityCounts", s.antiAffinityCounts.String())
			if tphash, hasTemplateHashInPod := pod.Annotations[apps.DefaultDeploymentUniqueLabelKey]; hasTemplateHashInPod {
				pl.filteringIncomingPodCache.impl.Write(tphash, NewFilteringIncomingPodAffinityTermDetailedState(
					s.affinityCounts, s.antiAffinityCounts, tpByPod, s.podInfo.GetRequiredAffinityTerms(), s.podInfo.GetRequiredAntiAffinityTerms()))
			}
		}
	}

	cycleState.Write(preFilterStateKey, s)
	return nil, nil
}

// PreFilterExtensions returns prefilter extensions, pod add and remove.
func (pl *InterPodAffinity) PreFilterExtensions() fwk.PreFilterExtensions {
	return pl
}

// AddPod from pre-computed data in cycleState.
func (pl *InterPodAffinity) AddPod(ctx context.Context, cycleState fwk.CycleState, podToSchedule *v1.Pod, podInfoToAdd fwk.PodInfo, nodeInfo fwk.NodeInfo) *fwk.Status {
	state, err := getPreFilterState(cycleState)
	if err != nil {
		return fwk.AsStatus(err)
	}
	state.updateWithPod(podInfoToAdd, nodeInfo.Node(), 1)
	return nil
}

// RemovePod from pre-computed data in cycleState.
func (pl *InterPodAffinity) RemovePod(ctx context.Context, cycleState fwk.CycleState, podToSchedule *v1.Pod, podInfoToRemove fwk.PodInfo, nodeInfo fwk.NodeInfo) *fwk.Status {
	state, err := getPreFilterState(cycleState)
	if err != nil {
		return fwk.AsStatus(err)
	}
	state.updateWithPod(podInfoToRemove, nodeInfo.Node(), -1)
	return nil
}

func getPreFilterState(cycleState fwk.CycleState) (*preFilterState, error) {
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
func satisfyExistingPodsAntiAffinity(state *preFilterState, nodeInfo fwk.NodeInfo) bool {
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

// Checks if the node satisfies the incoming pod's anti-affinity rules.
func satisfyPodAntiAffinity(state *preFilterState, nodeInfo fwk.NodeInfo) bool {
	if len(state.antiAffinityCounts) > 0 {
		for _, term := range state.podInfo.GetRequiredAntiAffinityTerms() {
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
func satisfyPodAffinity(state *preFilterState, nodeInfo fwk.NodeInfo) bool {
	podsExist := true
	for _, term := range state.podInfo.GetRequiredAffinityTerms() {
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
		if len(state.affinityCounts) == 0 && podMatchesAllAffinityTerms(state.podInfo.GetRequiredAffinityTerms(), state.podInfo.GetPod()) {
			return true
		}
		return false
	}
	return true
}

// Filter invoked at the filter extension point.
// It checks if a pod can be scheduled on the specified node with pod affinity/anti-affinity configuration.
func (pl *InterPodAffinity) Filter(ctx context.Context, cycleState fwk.CycleState, pod *v1.Pod, nodeInfo fwk.NodeInfo) *fwk.Status {

	state, err := getPreFilterState(cycleState)
	if err != nil {
		return fwk.AsStatus(err)
	}
	state.updateLock.RLock()
	defer state.updateLock.RUnlock()

	if !satisfyPodAffinity(state, nodeInfo) {
		return fwk.NewStatus(fwk.UnschedulableAndUnresolvable, ErrReasonAffinityRulesNotMatch)
	}

	if !satisfyPodAntiAffinity(state, nodeInfo) {
		return fwk.NewStatus(fwk.Unschedulable, ErrReasonAntiAffinityRulesNotMatch)
	}

	if !satisfyExistingPodsAntiAffinity(state, nodeInfo) {
		return fwk.NewStatus(fwk.Unschedulable, ErrReasonExistingAntiAffinityRulesNotMatch)
	}

	return nil
}
