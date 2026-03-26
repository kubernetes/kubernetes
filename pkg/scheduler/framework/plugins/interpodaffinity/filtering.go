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
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
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
	// existingAntiAffinityCounts is a map of topology pairs to the number of existing pods that has anti-affinity terms that match the pod.
	// It tracks existing pods' anti-affinity rules that the incoming pod violates.
	// This only considers non-host-scoped terms as host-scoped are checked locally in Filter.
	existingAntiAffinityCounts topologyToMatchedTermCount
	// A map of topology pairs to the number of existing pods that match the affinity terms of the pod.
	// affinityCounts and antiAffinityCounts track how many existing pods satisfy the incoming pod's non-host-scoped affinity terms.
	// For host-scoped affinity, we need to know if there's *any* pod globally that matches.
	// If not, and the pod has self-affinity, it might be schedulable on a node
	// where it's the first matching pod.
	affinityCounts topologyToMatchedTermCount
	// A map of topology pairs to the number of existing pods that match the anti-affinity terms of the pod.
	// antiAffinityCounts track how many existing pods satisfy the incoming pod's and anti-affinity terms.
	antiAffinityCounts topologyToMatchedTermCount
	// podInfo of the incoming pod.
	podInfo fwk.PodInfo
	// A copy of the incoming pod's namespace labels.
	namespaceLabels labels.Set

	// hostScopedAffinityTerms are affinity terms checked node-locally in Filter.
	// If an incoming pod has only hostname scoped affinity, we can skip expensive global topology checks.
	// hostScopedAffinityTerms is populated iff the incoming pod has only host-scoped affinity terms. Otherwise, this is empty, and allAffinityTerms is populated.
	// See the docstring for classifyTermsBasedOnScope for more details.
	hostScopedAffinityTerms []fwk.AffinityTerm
	// hostScopedAntiAffinityTerms are anti-affinity terms checked node-locally in Filter.
	// If an incoming pod has only hostname scoped anti-affinity, we can skip expensive global topology checks.
	hostScopedAntiAffinityTerms []fwk.AffinityTerm
	// allAffinityTerms contains both hostname and non-hostname scoped terms and is populated if a pod has any non-host-scoped terms.
	// If an incoming pod has only host-scoped affinity terms, allAffinityTerms will be empty.
	// See the docstring for classifyTermsBasedOnScope for more details.
	allAffinityTerms []fwk.AffinityTerm
	// nonHostScopedAntiAffinityTerms contains non-hostname scoped terms (e.g. zone).
	// If an incoming pod has only host-scoped anti-affinity terms, nonHostScopedAntiAffinityTerms will be empty.
	// See the docstring for classifyTermsBasedOnScope for more details.
	nonHostScopedAntiAffinityTerms []fwk.AffinityTerm

	hasMatchingHostScopedAffinityPodGlobally bool
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
	copy.allAffinityTerms = s.allAffinityTerms
	copy.nonHostScopedAntiAffinityTerms = s.nonHostScopedAntiAffinityTerms
	copy.hostScopedAffinityTerms = s.hostScopedAffinityTerms
	copy.hostScopedAntiAffinityTerms = s.hostScopedAntiAffinityTerms
	copy.hasMatchingHostScopedAffinityPodGlobally = s.hasMatchingHostScopedAffinityPodGlobally
	return &copy
}

// updateWithPod updates the preFilterState counters with the (anti)affinity matches for the given podInfo.
func (s *preFilterState) updateWithPod(pInfo fwk.PodInfo, node *v1.Node, multiplier int64) {
	if s == nil {
		return
	}

	s.existingAntiAffinityCounts.updateWithAntiAffinityTerms(pInfo.GetRequiredAntiAffinityTerms(), s.podInfo.GetPod(), s.namespaceLabels, node, multiplier)
	s.affinityCounts.updateWithAffinityTerms(s.allAffinityTerms, pInfo.GetPod(), node, multiplier)
	// The incoming pod's terms have the namespaceSelector merged into the namespaces, and so
	// here we don't lookup the updated pod's namespace labels, hence passing nil for nsLabels.
	s.antiAffinityCounts.updateWithAntiAffinityTerms(s.nonHostScopedAntiAffinityTerms, pInfo.GetPod(), nil, node, multiplier)
}

// classifyTermsBasedOnScope separates terms that can be checked purely locally on a single node from those that require a global scan across the cluster.
//
// For affinity, we are deciding if a candidate node is valid for the incoming pod to land on.
// A node becomes valid if it (or its wider topology, like a zone) already hosts pods that match the incoming pod's rules.
// The algorithm searches for a SINGLE existing pod in that topology that matches ALL of the incoming pod's affinity rules at once.
// Because we must find this "single multi-matching pod" in the wider topology, we cannot evaluate the terms independently; we must keep them grouped together in `allAffinityTerms`.
//
// For anti-affinity, we check with an OR of "Violated" ( Violated A OR Violated B) to determine if a node violates ANY anti-affinity constraints of the incoming pod.
// Regardless of other terms, hostname-scoped anti-affinity can always be checked locally to avoid global scans for those specific terms.
// For example, if a pod has both hostname and zone anti-affinity rules, we can reject the node immediately if it violates the hostname rule, without checking the zone rule.
// Since a violation of either is sufficient to reject a node, they don't need to be checked together.
//
// If a pod has only hostScopedAffinityTerms and only hostScopedAntiAffinityTerms, both allAffinityTerms and nonHostScopedAntiAffinityTerms will be empty.
// When that happens, pl.getIncomingAffinityAntiAffinityCounts exits early, skipping the expensive parallel loop over all nodes in the cluster.
func (s *preFilterState) classifyTermsBasedOnScope() {
	if s.allHostScoped() {
		s.hostScopedAffinityTerms = s.podInfo.GetRequiredAffinityTerms()
	} else {
		s.allAffinityTerms = s.podInfo.GetRequiredAffinityTerms()
	}

	for _, t := range s.podInfo.GetRequiredAntiAffinityTerms() {
		if t.TopologyKey == v1.LabelHostname {
			s.hostScopedAntiAffinityTerms = append(s.hostScopedAntiAffinityTerms, t)
		} else {
			s.nonHostScopedAntiAffinityTerms = append(s.nonHostScopedAntiAffinityTerms, t)
		}
	}
}

// allHostScoped determines if we can bypass expensive global topology scans. If all terms are host-scoped, we only need to check the local node's pods, which is much faster.
func (s *preFilterState) allHostScoped() bool {
	for _, t := range s.podInfo.GetRequiredAffinityTerms() {
		if t.TopologyKey != v1.LabelHostname {
			return false
		}
	}
	return true
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

func (m *topologyToMatchedTermCountList) append(node *v1.Node, tk string, value int64) {
	if tv, ok := node.Labels[tk]; ok {
		pair := topologyPair{key: tk, value: tv}
		*m = append(*m, topologyPairCount{
			topologyPair: pair,
			count:        value,
		})
	}
}

// appends the specified value to the topologyToMatchedTermCountList
// for each affinity term if "targetPod" matches ALL terms.
func (m *topologyToMatchedTermCountList) appendWithAffinityTerms(
	terms []fwk.AffinityTerm, pod *v1.Pod, node *v1.Node, value int64) {
	if podMatchesAllAffinityTerms(terms, pod) {
		for _, t := range terms {
			m.append(node, t.TopologyKey, value)
		}
	}
}

// appends the specified value to the topologyToMatchedTermCountList
// for each anti-affinity term matched the target pod.
func (m *topologyToMatchedTermCountList) appendWithAntiAffinityTerms(terms []fwk.AffinityTerm, pod *v1.Pod, nsLabels labels.Set, node *v1.Node, value int64) {
	// Check anti-affinity terms.
	for _, t := range terms {
		if t.Matches(pod, nsLabels) {
			m.append(node, t.TopologyKey, value)
		}
	}
}

// podMatchesAllAffinityTerms returns true if the given terms list is not empty and given pod matches all the given terms.
// Useful for checking affinity. For anti-affinity, use podMatchesAnyAffinityTerms.
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

// podMatchesAnyAffinityTerms returns true if the given pod matches any of the given terms.
// Useful for checking anti-affinity.
func podMatchesAnyAffinityTerms(terms []fwk.AffinityTerm, pod *v1.Pod) bool {
	for _, t := range terms {
		if t.Matches(pod, nil) {
			return true
		}
	}
	return false
}

// calculates the following for each existing pod on each node:
//  1. Whether it has PodAntiAffinity
//  2. Whether any AntiAffinityTerm matches the incoming pod
func (pl *InterPodAffinity) getExistingAntiAffinityCounts(ctx context.Context, pod *v1.Pod, nsLabels labels.Set, nodes []fwk.NodeInfo) (topologyToMatchedTermCount, bool) {
	antiAffinityCountsList := make([]topologyToMatchedTermCountList, len(nodes))
	index := int32(-1)
	var hasHostScopedAntiAffinity int32
	processNode := func(i int) {
		nodeInfo := nodes[i]
		node := nodeInfo.Node()

		antiAffinityCounts := make(topologyToMatchedTermCountList, 0)
		for _, existingPod := range nodeInfo.GetPodsWithRequiredAntiAffinity() {
			for _, term := range existingPod.GetRequiredAntiAffinityTerms() {
				if term.TopologyKey == v1.LabelHostname {
					atomic.StoreInt32(&hasHostScopedAntiAffinity, 1)
				} else if term.Matches(pod, nsLabels) {
					antiAffinityCounts.append(node, term.TopologyKey, 1)
				}
			}
		}
		if len(antiAffinityCounts) != 0 {
			antiAffinityCountsList[atomic.AddInt32(&index, 1)] = antiAffinityCounts
		}
	}
	pl.parallelizer.Until(ctx, len(nodes), processNode, pl.Name())

	result := make(topologyToMatchedTermCount)
	// Traditional for loop is slightly faster in this case than its "for range" equivalent.
	for i := 0; i <= int(index); i++ {
		result.mergeWithList(antiAffinityCountsList[i])
	}

	return result, hasHostScopedAntiAffinity == 1
}

// finds existing Pods that match affinity terms of the incoming pod's (anti)affinity terms.
// It returns a topologyToMatchedTermCount that are checked later by the affinity
// predicate. With this topologyToMatchedTermCount available, the affinity predicate does not
// need to check all the pods in the cluster.
func (pl *InterPodAffinity) getIncomingAffinityAntiAffinityCounts(ctx context.Context, affinityTerms, antiAffinityTerms []fwk.AffinityTerm, allNodes []fwk.NodeInfo) (topologyToMatchedTermCount, topologyToMatchedTermCount) {
	affinityCounts := make(topologyToMatchedTermCount)
	antiAffinityCounts := make(topologyToMatchedTermCount)
	if len(affinityTerms) == 0 && len(antiAffinityTerms) == 0 {
		return affinityCounts, antiAffinityCounts
	}

	affinityCountsList := make([]topologyToMatchedTermCountList, len(allNodes))
	antiAffinityCountsList := make([]topologyToMatchedTermCountList, len(allNodes))
	index := int32(-1)
	processNode := func(i int) {
		nodeInfo := allNodes[i]
		node := nodeInfo.Node()

		affinity := make(topologyToMatchedTermCountList, 0)
		antiAffinity := make(topologyToMatchedTermCountList, 0)
		for _, existingPod := range nodeInfo.GetPods() {
			affinity.appendWithAffinityTerms(affinityTerms, existingPod.GetPod(), node, 1)
			// The incoming pod's terms have the namespaceSelector merged into the namespaces, and so
			// here we don't lookup the existing pod's namespace labels, hence passing nil for nsLabels.
			antiAffinity.appendWithAntiAffinityTerms(antiAffinityTerms, existingPod.GetPod(), nil, node, 1)
		}

		if len(affinity) > 0 || len(antiAffinity) > 0 {
			k := atomic.AddInt32(&index, 1)
			affinityCountsList[k] = affinity
			antiAffinityCountsList[k] = antiAffinity
		}
	}
	pl.parallelizer.Until(ctx, len(allNodes), processNode, pl.Name())

	for i := 0; i <= int(index); i++ {
		affinityCounts.mergeWithList(affinityCountsList[i])
		antiAffinityCounts.mergeWithList(antiAffinityCountsList[i])
	}

	return affinityCounts, antiAffinityCounts
}

// PreFilter invoked at the prefilter extension point. Pre-computes counts of
// existing pods that satisfy/conflict with the incoming pod's rules. It separates
// hostname-scoped terms (which allow fast-path local checks) from wider topology checks.
// Host-scoped terms have a TopologyKey of v1.LabelHostname.
func (pl *InterPodAffinity) PreFilter(ctx context.Context, cycleState fwk.CycleState, pod *v1.Pod, allNodes []fwk.NodeInfo) (*fwk.PreFilterResult, *fwk.Status) {
	var nodesWithNonHostScopedAntiAffinityPods []fwk.NodeInfo
	var nodesWithAnyRequiredAntiAffinityPods []fwk.NodeInfo
	var err error
	if nodesWithNonHostScopedAntiAffinityPods, err = pl.sharedLister.NodeInfos().HavePodsWithRequiredNonHostScopedAntiAffinityList(); err != nil {
		return nil, fwk.AsStatus(fmt.Errorf("failed to list NodeInfos with pods with topology-scoped anti-affinity: %w", err))
	}
	if nodesWithAnyRequiredAntiAffinityPods, err = pl.sharedLister.NodeInfos().HavePodsWithRequiredAntiAffinityList(); err != nil {
		return nil, fwk.AsStatus(fmt.Errorf("failed to list NodeInfos with pods with required anti-affinity: %w", err))
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

	s.classifyTermsBasedOnScope()

	logger := klog.FromContext(ctx)
	s.namespaceLabels = GetNamespaceLabelsSnapshot(logger, pod.Namespace, pl.nsLister)

	var hasHostScopedInCounts bool
	s.existingAntiAffinityCounts, hasHostScopedInCounts = pl.getExistingAntiAffinityCounts(ctx, pod, s.namespaceLabels, nodesWithNonHostScopedAntiAffinityPods)
	hasHostScopedAntiAffinity := hasHostScopedInCounts || len(nodesWithAnyRequiredAntiAffinityPods) > len(nodesWithNonHostScopedAntiAffinityPods)

	s.affinityCounts, s.antiAffinityCounts = pl.getIncomingAffinityAntiAffinityCounts(ctx, s.allAffinityTerms, s.nonHostScopedAntiAffinityTerms, allNodes)

	// Check if a pod with mastching host-scoped affinity exist on the cluster
	if len(s.hostScopedAffinityTerms) > 0 {
		s.hasMatchingHostScopedAffinityPodGlobally = hasMatchingHostScopedAffinityPodGlobally(ctx, allNodes, s, pl)
	}

	if len(s.existingAntiAffinityCounts) == 0 && !hasHostScopedAntiAffinity && len(s.podInfo.GetRequiredAffinityTerms()) == 0 && len(s.podInfo.GetRequiredAntiAffinityTerms()) == 0 {
		return nil, fwk.NewStatus(fwk.Skip)
	}

	cycleState.Write(preFilterStateKey, s)
	return nil, nil
}

func hasMatchingHostScopedAffinityPodGlobally(ctx context.Context, allNodes []fwk.NodeInfo, s *preFilterState, pl *InterPodAffinity) bool {
	var found int32
	cancelCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	processNode := func(i int) {
		if atomic.LoadInt32(&found) != 0 {
			return
		}
		nodeInfo := allNodes[i]
		for _, existingPod := range nodeInfo.GetPods() {
			if podMatchesAllAffinityTerms(s.hostScopedAffinityTerms, existingPod.GetPod()) {
				if atomic.CompareAndSwapInt32(&found, 0, 1) {
					cancel()
				}
				return
			}
		}
	}
	pl.parallelizer.Until(cancelCtx, len(allNodes), processNode, pl.Name())
	return atomic.LoadInt32(&found) != 0
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

	// Local check for host-scoped anti-affinity
	for _, existingPodInfo := range nodeInfo.GetPodsWithRequiredAntiAffinity() {
		for _, term := range existingPodInfo.GetRequiredAntiAffinityTerms() {
			if term.TopologyKey == v1.LabelHostname {
				if term.Matches(state.podInfo.GetPod(), state.namespaceLabels) {
					return false
				}
			}
		}
	}

	return true
}

// Checks if the node satisfies the incoming pod's anti-affinity rules.
// nodeTopologyConflictsWithAntiAffinity returns true if the node's wider topology domain already houses conflicting pods, based on pre-computed global state.
func nodeTopologyConflictsWithAntiAffinity(state *preFilterState, nodeInfo fwk.NodeInfo) bool {
	if len(state.antiAffinityCounts) > 0 {
		for _, term := range state.nonHostScopedAntiAffinityTerms {
			if topologyValue, ok := nodeInfo.Node().Labels[term.TopologyKey]; ok {
				tp := topologyPair{key: term.TopologyKey, value: topologyValue}
				if state.antiAffinityCounts[tp] > 0 {
					return true
				}
			}
		}
	}
	return false
}

// nodePodsConflictWithAntiAffinity returns true if the target node already hosts pods that conflict with the incoming pod's host-scoped anti-affinity rules.
func nodePodsConflictWithAntiAffinity(state *preFilterState, nodeInfo fwk.NodeInfo) bool {
	for _, term := range state.hostScopedAntiAffinityTerms {
		for _, existingPod := range nodeInfo.GetPods() {
			if term.Matches(existingPod.GetPod(), nil) {
				return true
			}
		}
	}
	return false
}

// nodePodsSatisfyAffinity performs the fast-path check for host-scoped affinity terms by inspecting pods already on the target node. It also handles the "first pod of its kind" self-affinity scenario, allowing a pod to land on an empty node if it matches its own terms.
func nodePodsSatisfyAffinity(state *preFilterState, nodeInfo fwk.NodeInfo) bool {
	if len(state.hostScopedAffinityTerms) > 0 {
		matchFound := false
		for _, existingPod := range nodeInfo.GetPods() {
			if podMatchesAllAffinityTerms(state.hostScopedAffinityTerms, existingPod.GetPod()) {
				matchFound = true
				break
			}
		}
		if !matchFound {
			return satisfiesSelfAffinityCheck(state, nodeInfo)
		}
		return true
	}
	return true
}

// nodeTopologyMatchesAffinity verifies if the node's topology domain contains at least one pod matching the affinity terms, or if the pod can be the first of its kind to land on this node (self-affinity). This path is executed when not all affinity terms are host-scoped, requiring us to check pre-computed global state.
func nodeTopologyMatchesAffinity(state *preFilterState, nodeInfo fwk.NodeInfo) bool {
	if len(state.allAffinityTerms) > 0 {
		podsExist := true
		for _, term := range state.allAffinityTerms {
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
			return len(state.affinityCounts) == 0 && satisfiesSelfAffinityCheck(state, nodeInfo)
		}
		return true
	}
	return true
}

// This pod may be the first pod in a series that have affinity to themselves. In order
// to not leave such pods in pending state forever, we check that if no other pod
// in the cluster matches the namespace and selector of this pod, the pod matches
// its own terms, and the node has all the requested topologies, then we allow the pod
// to pass the affinity check.
func satisfiesSelfAffinityCheck(state *preFilterState, nodeInfo fwk.NodeInfo) bool {
	if !state.hasMatchingHostScopedAffinityPodGlobally && podMatchesAllAffinityTerms(state.podInfo.GetRequiredAffinityTerms(), state.podInfo.GetPod()) {
		for _, term := range state.podInfo.GetRequiredAffinityTerms() {
			if _, ok := nodeInfo.Node().Labels[term.TopologyKey]; !ok {
				return false
			}
		}
		return true
	}
	return false
}

// Filter invoked at the filter extension point.
// It checks if a pod can be scheduled on the specified node with pod affinity/anti-affinity configuration.
func (pl *InterPodAffinity) Filter(ctx context.Context, cycleState fwk.CycleState, pod *v1.Pod, nodeInfo fwk.NodeInfo) *fwk.Status {

	state, err := getPreFilterState(cycleState)
	if err != nil {
		return fwk.AsStatus(err)
	}

	if len(state.hostScopedAffinityTerms) > 0 {
		if !nodePodsSatisfyAffinity(state, nodeInfo) {
			return fwk.NewStatus(fwk.UnschedulableAndUnresolvable, ErrReasonAffinityRulesNotMatch)
		}
	} else if len(state.allAffinityTerms) > 0 {
		if !nodeTopologyMatchesAffinity(state, nodeInfo) {
			return fwk.NewStatus(fwk.UnschedulableAndUnresolvable, ErrReasonAffinityRulesNotMatch)
		}
	}

	if nodeTopologyConflictsWithAntiAffinity(state, nodeInfo) || nodePodsConflictWithAntiAffinity(state, nodeInfo) {
		return fwk.NewStatus(fwk.Unschedulable, ErrReasonAntiAffinityRulesNotMatch)
	}

	if !satisfyExistingPodsAntiAffinity(state, nodeInfo) {
		return fwk.NewStatus(fwk.Unschedulable, ErrReasonExistingAntiAffinityRulesNotMatch)
	}

	return nil
}
