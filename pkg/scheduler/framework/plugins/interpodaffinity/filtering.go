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
	"k8s.io/component-helpers/resource"
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

// preFilterState is computed at PreFilter and used at Filter.
//
// Overview of InterPodAffinityHostnameFastPath optimization:
// The InterPodAffinityHostnameFastPath feature optimizes performance of PreFilter and Filter for required
// (anti) affinity that exclusively uses the kubernetes.io/hostname topology.
//
// In the general case, an existing pod anywhere in the cluster can affect whether an incoming pod
// schedules on a particular node (for example, if they need to share the same "zone"). To handle this,
// the standard algorithm performs a heavy, cluster-wide topology map construction during PreFilter to
// track matching pods across all topologies. During Filter, the scheduler must then iterate over every label
// on a candidate node and perform hash map lookups to verify constraints against this global map.
//
// However, when rules use the `kubernetes.io/hostname` topology key, they only care about pods on that
// exact local node. By decoupling hostname-scoped rules, we can completely bypass the expensive topology
// map construction in PreFilter and the node-label iteration loops in Filter.
//
// How this works in more detail:
//
//  1. Existing pods' Anti-Affinity:
//     Existing pods may have anti-affinity terms that reject the incoming pod. Hostname-scoped terms from
//     existing pods are never added to the global topology map (existingClusterWideAntiAffinityCounts).
//     In Filter, the expensive node-label map lookups are skipped, falling back to a cheap local check against the candidate node's pods.
//
//  2. Incoming pod's Anti-Affinity:
//     Likewise, the incoming pod's own hostname-scoped anti-affinity terms bypass global map construction
//     and are deferred to a cheap local check during Filter.
//
//  3. Incoming pod's Affinity:
//     Affinity is trickier because it requires finding a single existing pod that matches ALL affinity terms at once.
//     - Mixed Topology: If the incoming pod has at least one non-hostname-scoped affinity term, PreFilter
//     must construct the global clusterWideAffinityCounts map to correctly track matches across all topology domains.
//     - Exclusively Hostname-Scoped: If ALL affinity terms on the incoming pod are hostname-scoped, we
//     completely bypass building the global affinity map. Affinity is verified entirely during the local Filter phase.
//
//  4. Self-Affinity Fallback (matchingHostScopedAffinityPodsCount):
//     If no existing pods match the incoming pod's affinity rules, we still allow the pod to schedule if it
//     satisfies its own affinity terms (self-affinity). Normally, we know no other matching pods exist because
//     the global map (clusterWideAffinityCounts) remains empty. But if we bypass building the global map
//     (as described in #3), we lose that cluster-wide visibility. To fix this, instead of building the map
//     during PreFilter, we maintain a counter (`matchingHostScopedAffinityPodsCount`)
//     just to track if ANY matching pods exist anywhere. If this counter remains exactly zero, we can allow the self-matching pod to schedule.

//  5. PreFilterExtensions (Preemption / AddPod / RemovePod):
//     When the scheduler simulates evicting pods for preemption, it avoids rebuilding state from scratch by
//     incrementally updating the PreFilter state (via AddPod/RemovePod). If an incoming pod has only host-scoped terms,
//     simulating the removal or addition of pod becomes a lightweight increment or decrement of a counter (matchingHostScopedAffinityPodsCount) instead
//     a global topology map update.

type preFilterState struct {
	// existingClusterWideAntiAffinityCounts is a map of topology pairs to the number of existing pods that have anti-affinity terms matching the incoming pod.
	// It tracks existing pods' anti-affinity rules that the incoming pod violates.
	// If the InterPodAffinityHostnameFastPath feature gate is enabled, this only considers non-hostname-scoped terms (hostname-scoped are checked locally in Filter).
	// If the feature gate is disabled, this considers all terms (both hostname and non-hostname scoped).
	existingClusterWideAntiAffinityCounts topologyToMatchedTermCount
	// clusterWideAffinityCounts is a map of topology pairs to the number of existing pods that match the incoming pod's affinity terms.
	// If the InterPodAffinityHostnameFastPath feature gate is enabled this tracks matches for the terms in clusterWideAffinityTerms (which includes both hostname and non-hostname scoped terms if ANY non-hostname scoped term exists, and is empty otherwise).
	// If the feature gate is disabled, this tracks matches for all affinity terms.
	clusterWideAffinityCounts topologyToMatchedTermCount
	// clusterWideAntiAffinityCounts is a map of topology pairs to the number of existing pods that match the incoming pod's anti-affinity terms.
	// If the InterPodAffinityHostnameFastPath feature gate is enabled, this tracks matches only for non-hostname-scoped terms.
	// If the feature gate is disabled, this tracks matches for all anti-affinity terms.
	clusterWideAntiAffinityCounts topologyToMatchedTermCount
	// podInfo of the incoming pod.
	podInfo fwk.PodInfo
	// A copy of the incoming pod's namespace labels.
	namespaceLabels labels.Set

	// hostScopedAffinityTerms are the incoming pod's affinity terms that will be checked node-locally in Filter.
	// If the InterPodAffinityHostnameFastPath feature gate is enabled, this is populated ONLY if the incoming pod has exclusively hostname-scoped affinity terms.
	// If the feature gate is disabled, this remains empty.
	hostScopedAffinityTerms []fwk.AffinityTerm
	// hostScopedAntiAffinityTerms are the incoming pod's anti-affinity terms that will be checked node-locally in Filter.
	// If the InterPodAffinityHostnameFastPath feature gate is enabled, this contains all hostname-scoped anti-affinity terms from the incoming pod.
	// If the feature gate is disabled, this remains empty.
	hostScopedAntiAffinityTerms []fwk.AffinityTerm
	// clusterWideAffinityTerms contains affinity terms from the incoming pod that require a global cluster scan.
	// If the InterPodAffinityHostnameFastPath feature gate is enabled: it is populated with ALL affinity terms from the incoming pod (both hostname and non-hostname scoped), if the incoming pod has at least one non-hostname-scoped affinity term. It is empty if the incoming pod has ONLY hostname-scoped affinity terms.
	// If the feature gate is disabled: it contains the incoming pod's ALL affinity terms regardless of their scope.
	clusterWideAffinityTerms []fwk.AffinityTerm
	// clusterWideAntiAffinityTerms contains anti-affinity terms from the incoming pod that require a global cluster scan.
	// If the InterPodAffinityHostnameFastPath feature gate is enabled: it contains only the non-hostname-scoped anti-affinity terms from the incoming pod.
	// If the feature gate is disabled: it contains ALL anti-affinity terms (both hostname and non-hostname scoped) from the incoming pod.
	clusterWideAntiAffinityTerms []fwk.AffinityTerm
	// matchingHostScopedAffinityPodsCount is the total number of pods in the cluster that match the incoming pod's host-scoped affinity terms.
	// It is used to optimize checks when all affinity terms in the incoming pod are host-scoped.
	// This will be 0 if the InterPodAffinityHostnameFastPath feature gate is disabled.
	matchingHostScopedAffinityPodsCount int64
	// enableInterPodAffinityHostnameFastPath is true if the InterPodAffinityHostnameFastPath feature gate is enabled.
	enableInterPodAffinityHostnameFastPath bool
}

// Clone the prefilter state.
func (s *preFilterState) Clone() fwk.StateData {
	if s == nil {
		return nil
	}

	copy := preFilterState{}
	copy.clusterWideAffinityCounts = s.clusterWideAffinityCounts.clone()
	copy.clusterWideAntiAffinityCounts = s.clusterWideAntiAffinityCounts.clone()
	copy.existingClusterWideAntiAffinityCounts = s.existingClusterWideAntiAffinityCounts.clone()
	// No need to deep copy the podInfo because it shouldn't change.
	copy.podInfo = s.podInfo
	copy.namespaceLabels = s.namespaceLabels
	copy.clusterWideAffinityTerms = s.clusterWideAffinityTerms
	copy.clusterWideAntiAffinityTerms = s.clusterWideAntiAffinityTerms
	copy.hostScopedAffinityTerms = s.hostScopedAffinityTerms
	copy.hostScopedAntiAffinityTerms = s.hostScopedAntiAffinityTerms
	copy.matchingHostScopedAffinityPodsCount = s.matchingHostScopedAffinityPodsCount
	copy.enableInterPodAffinityHostnameFastPath = s.enableInterPodAffinityHostnameFastPath
	return &copy
}

// updateWithPod updates the preFilterState counters with the (anti)affinity matches for the given podInfo.
func (s *preFilterState) updateWithPod(pInfo fwk.PodInfo, node *v1.Node, multiplier int64) {
	if s == nil {
		return
	}

	// Process the existing pod's anti-affinity terms to see if the incoming pod violates them.
	if s.enableInterPodAffinityHostnameFastPath {
		// We iterate over the terms and manually check the topology key on the fly because
		// the existing pod's terms are not pre-classified by scope.
		for _, t := range pInfo.GetRequiredAntiAffinityTerms() {
			if t.TopologyKey != v1.LabelHostname {
				if t.Matches(s.podInfo.GetPod(), s.namespaceLabels) {
					s.existingClusterWideAntiAffinityCounts.update(node, t.TopologyKey, multiplier)
				}
			}
		}
	} else {
		s.existingClusterWideAntiAffinityCounts.updateWithAntiAffinityTerms(pInfo.GetRequiredAntiAffinityTerms(), s.podInfo.GetPod(), s.namespaceLabels, node, multiplier)
	}
	// Process the incoming pod's affinity and anti-affinity terms.
	if s.enableInterPodAffinityHostnameFastPath {
		// Here we only evaluate the terms in s.clusterWideAffinityTerms and s.clusterWideAntiAffinityTerms as those require global evaluation.
		// Unlike for the existing pods, we do not need to manually check the topology key for incoming pod's terms because they have been preprocessed by classifyTermsBasedOnScope() during PreFilter.
		s.clusterWideAffinityCounts.updateWithAffinityTerms(s.clusterWideAffinityTerms, pInfo.GetPod(), node, multiplier)
		// The incoming pod's terms have the namespaceSelector merged into the namespaces, and so
		// here we don't lookup the updated pod's namespace labels, hence passing nil for nsLabels.
		s.clusterWideAntiAffinityCounts.updateWithAntiAffinityTerms(s.clusterWideAntiAffinityTerms, pInfo.GetPod(), nil, node, multiplier)

		if len(s.hostScopedAffinityTerms) > 0 {
			// hostScopedAffinityTerms only gets populated if ALL affinity terms of the incoming pod are host-scoped.
			//
			// This plugin keeps a running global counter of all matching pods in the cluster.
			// This counter is populated and updated iff the incoming pod has ONLY host-scoped affinity terms.
			//
			// If this counter is exactly 0, the plugin knows no existing pods satisfy the rules, permitting the first pod in a self-affinity group to pass this affinity check.
			// (If the pod has any non-host scoped affinity terms, this counter remains 0 and the plugin checks 'len(clusterWideAffinityCounts) == 0' instead).
			//
			// The multiplier (+1 or -1) adjusts this counter during "what-if" scheduling simulations like preemption.
			if podMatchesAllAffinityTerms(s.hostScopedAffinityTerms, pInfo.GetPod()) {
				s.matchingHostScopedAffinityPodsCount += multiplier
			}
		}
	} else {
		s.clusterWideAffinityCounts.updateWithAffinityTerms(s.podInfo.GetRequiredAffinityTerms(), pInfo.GetPod(), node, multiplier)
		// The incoming pod's terms have the namespaceSelector merged into the namespaces, and so
		// here we don't lookup the updated pod's namespace labels, hence passing nil for nsLabels.
		s.clusterWideAntiAffinityCounts.updateWithAntiAffinityTerms(s.podInfo.GetRequiredAntiAffinityTerms(), pInfo.GetPod(), nil, node, multiplier)
	}
}

// classifyTermsBasedOnScope groups the (anti-)affinity terms of the incoming pod into lists:
// - clusterWideAffinityTerms - terms that need to be evaluated in a global check (across the entire cluster)
// - clusterWideAntiAffinityTerms - terms that need to be evaluated in a global check
// - hostScopedAffinityTerms - terms that can be evaluated locally on each node, skipping the expensive global scan
// Note that a term may be in more than one list at a time.
//
// For affinity, we are deciding if a candidate node is valid for the incoming pod to land on.
// A node becomes valid if it (or its wider topology, like a zone) already hosts pods that match the incoming pod's rules.
// The algorithm searches for a SINGLE existing pod in that topology that matches ALL of the incoming pod's affinity rules at once.
// Because we must find this "single multi-matching pod" in the wider topology, we cannot evaluate the host-scoped and non-host-scoped terms independently; we must keep them grouped together in `clusterWideAffinityTerms`.
//
// For anti-affinity, we check with an OR of "Violated" (Violated A OR Violated B) to determine if a node violates ANY anti-affinity constraints of the incoming pod.
// Regardless of other terms, hostname-scoped anti-affinity can always be checked locally to avoid global scans for those specific terms.
// For example, if a pod has both hostname and zone anti-affinity rules, we can reject the node immediately if it violates the hostname rule, without checking the zone rule.
// Since a violation of either is sufficient to reject a node, they don't need to be checked together.
//
// If a pod has only hostScopedAffinityTerms and only hostScopedAntiAffinityTerms, both clusterWideAffinityTerms and clusterWideAntiAffinityTerms will be empty.
// When that happens, pl.getIncomingAffinityAntiAffinityCounts exits early, skipping the expensive parallel loop over all nodes in the cluster.
func (s *preFilterState) classifyTermsBasedOnScope() {
	if s.allHostScoped() {
		s.hostScopedAffinityTerms = s.podInfo.GetRequiredAffinityTerms()
	} else {
		s.clusterWideAffinityTerms = s.podInfo.GetRequiredAffinityTerms()
	}

	for _, t := range s.podInfo.GetRequiredAntiAffinityTerms() {
		if t.TopologyKey == v1.LabelHostname {
			s.hostScopedAntiAffinityTerms = append(s.hostScopedAntiAffinityTerms, t)
		} else {
			s.clusterWideAntiAffinityTerms = append(s.clusterWideAntiAffinityTerms, t)
		}
	}
}

// allHostScoped determines if we can bypass expensive global topology scans. If all affinity terms are host-scoped, we only need to check the local node's pods, which is much faster.
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

func (m *topologyToMatchedTermCountList) recordMatch(node *v1.Node, tk string, value int64) {
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
			m.recordMatch(node, t.TopologyKey, value)
		}
	}
}

// appends the specified value to the topologyToMatchedTermCountList
// for each anti-affinity term matched the target pod.
func (m *topologyToMatchedTermCountList) appendWithAntiAffinityTerms(terms []fwk.AffinityTerm, pod *v1.Pod, nsLabels labels.Set, node *v1.Node, value int64) {
	// Check anti-affinity terms.
	for _, t := range terms {
		if t.Matches(pod, nsLabels) {
			m.recordMatch(node, t.TopologyKey, value)
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

// getExistingAntiAffinityCounts evaluates anti-affinity terms of existing pods
// to see if the incoming pod matches them. It executes a concurrent scan over nodes to count instances where the incoming pod creates a scheduling conflict.
//
// Returns two values:
//
//  1. topologyToMatchedTermCount: A map of topologies to the number of existing pods whose anti-affinity terms match the incoming pod.
//     - If InterPodAffinityHostnameFastPath is ENABLED: Only non-hostname scoped terms are tallied here (host-scoped terms are evaluated dynamically during Filter).
//     - If DISABLED: All matched anti-affinity terms are counted here.
//
//  2. hasHostScopedAntiAffinity (bool): Whether any evaluated existing pod has a hostname-scoped anti-affinity term.
//     - If InterPodAffinityHostnameFastPath is ENABLED: Returns true if a hostname-scoped term  was encountered. This indicates that a node-local check is required in Filter.
//     - If DISABLED: Always returns false.
//
// If hasHostScopedAntiAffinity is false and the count map is empty, the Filter phase can be skipped.
func (pl *InterPodAffinity) getExistingAntiAffinityCounts(ctx context.Context, incomingPod *v1.Pod, nsLabels labels.Set, nodes []fwk.NodeInfo) (topologyToMatchedTermCount, bool) {
	antiAffinityCountsList := make([]topologyToMatchedTermCountList, len(nodes))
	index := int32(-1)
	var hasHostScopedAntiAffinity atomic.Bool
	processNode := func(i int) {
		nodeInfo := nodes[i]
		node := nodeInfo.Node()

		clusterWideAntiAffinityCounts := make(topologyToMatchedTermCountList, 0)
		hasHostScoped := false
		for _, existingPod := range nodeInfo.GetPodsWithRequiredAntiAffinity() {
			for _, term := range existingPod.GetRequiredAntiAffinityTerms() {
				if pl.enableInterPodAffinityHostnameFastPath && term.TopologyKey == v1.LabelHostname {
					hasHostScoped = true
				} else if term.Matches(incomingPod, nsLabels) {
					clusterWideAntiAffinityCounts.recordMatch(node, term.TopologyKey, 1)
				}
			}
		}
		if hasHostScoped && !hasHostScopedAntiAffinity.Load() {
			hasHostScopedAntiAffinity.Store(true)
		}
		if len(clusterWideAntiAffinityCounts) != 0 {
			antiAffinityCountsList[atomic.AddInt32(&index, 1)] = clusterWideAntiAffinityCounts
		}
	}
	pl.parallelizer.Until(ctx, len(nodes), processNode, pl.Name())

	result := make(topologyToMatchedTermCount)
	// Traditional for loop is slightly faster in this case than its "for range" equivalent.
	for i := 0; i <= int(index); i++ {
		result.mergeWithList(antiAffinityCountsList[i])
	}

	return result, hasHostScopedAntiAffinity.Load()
}

// getIncomingAffinityAntiAffinityCounts scans all nodes concurrently to count how many
// existing pods match the incoming pod's affinity and anti-affinity terms.
//
// Returns two maps (topology -> matching pod count):
// 1. Affinity matches (existing pods satisfying the incoming pod's constraints).
// 2. Anti-affinity matches (existing pods violating the incoming pod's constraints).
//
// Note: If the InterPodAffinityHostnameFastPath feature is enabled, the caller passes
// pre-filtered term lists containing only the terms that require global evaluation (see docstring for classifyTermsBasedOnScope).
func (pl *InterPodAffinity) getIncomingAffinityAntiAffinityCounts(ctx context.Context, affinityTerms, antiAffinityTerms []fwk.AffinityTerm, allNodes []fwk.NodeInfo) (topologyToMatchedTermCount, topologyToMatchedTermCount) {
	clusterWideAffinityCounts := make(topologyToMatchedTermCount)
	clusterWideAntiAffinityCounts := make(topologyToMatchedTermCount)
	if len(affinityTerms) == 0 && len(antiAffinityTerms) == 0 {
		return clusterWideAffinityCounts, clusterWideAntiAffinityCounts
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
		clusterWideAffinityCounts.mergeWithList(affinityCountsList[i])
		clusterWideAntiAffinityCounts.mergeWithList(antiAffinityCountsList[i])
	}

	return clusterWideAffinityCounts, clusterWideAntiAffinityCounts
}

// PreFilter is invoked at the prefilter extension point to pre-compute state for the node-by-node Filter phase.
// Its primary job is to evaluate the incoming pod against existing pods in the cluster to build
// hash maps mapping topology pairs to match counts used to enforce affinity and anti-affinity.
//
// If the InterPodAffinityHostnameFastPath feature gate is DISABLED:
//  1. Existing Pods' Anti-Affinity: Queries the cache for the subset of nodes housing pods with ANY required
//     anti-affinity. Iterates over all pods on ONLY these nodes to populate `existingClusterWideAntiAffinityCounts`,
//     which tracks topologies that reject the incoming pod.
//  2. Incoming Pod's Constraints: Iterates over ALL nodes and ALL pods in the cluster, evaluating whether each
//     existing pod matches the incoming pod's anti-affinity and affinity terms, populating the
//     `clusterWideAntiAffinityCounts` and `clusterWideAffinityCounts` global maps.
//
// If the InterPodAffinityHostnameFastPath feature gate is ENABLED:
// The logic reduces the size of the maps by ignoring exactly `kubernetes.io/hostname` (host-scoped) terms.
//  1. Existing Pods' Anti-Affinity: Queries the cache for a narrower subset of nodes: ONLY nodes housing pods
//     with NON-host-scoped anti-affinity. It iterates over pods on these nodes to build the
//     `existingClusterWideAntiAffinityCounts` map. It also compares this node array length against the total
//     anti-affinity node array length to determine if host-scoped anti-affinity checks are needed later in Filter.
//  2. Incoming Pod's Constraints: Categorizes the incoming pod's terms into host-scoped and cluster-wide lists.
//     It iterates over ALL nodes and ALL pods in the cluster, evaluating whether each existing pod matches the
//     non-host-scoped terms on the incoming pod, building maps exclusively for those terms.
//  3. Self-Affinity Fallback: If the incoming pod has exclusively host-scoped affinity terms, no map is built.
//     Instead, PreFilter triggers a fast parallel traversal over ALL nodes and ALL pods to count the number of matching pods in
//     `matchingHostScopedAffinityPodsCount`.
//
// Finally, if no global tracking maps were populated, no host-scoped evaluation is needed, and the incoming pod
// has no terms of its own, PreFilter returns `fwk.Skip` to completely bypass this plugin during the Filter phase.
func (pl *InterPodAffinity) PreFilter(ctx context.Context, cycleState fwk.CycleState, pod *v1.Pod, allNodes []fwk.NodeInfo) (*fwk.PreFilterResult, *fwk.Status) {
	if pl.enableInPlacePodVerticalScalingSchedulerPreemption && resource.IsPodResizeDeferred(pod) {
		return nil, fwk.NewStatus(fwk.Skip)
	}
	s := &preFilterState{
		enableInterPodAffinityHostnameFastPath: pl.enableInterPodAffinityHostnameFastPath,
	}
	var err error
	if s.podInfo, err = framework.NewPodInfo(pod); err != nil {
		return nil, fwk.NewStatus(fwk.UnschedulableAndUnresolvable, fmt.Sprintf("parsing pod: %+v", err))
	}

	requiredAffinityTerms := s.podInfo.GetRequiredAffinityTerms()
	for i := range requiredAffinityTerms {
		if err := pl.mergeAffinityTermNamespacesIfNotEmpty(&requiredAffinityTerms[i]); err != nil {
			return nil, fwk.AsStatus(err)
		}
	}
	requiredAntiAffinityTerms := s.podInfo.GetRequiredAntiAffinityTerms()
	for i := range requiredAntiAffinityTerms {
		if err := pl.mergeAffinityTermNamespacesIfNotEmpty(&requiredAntiAffinityTerms[i]); err != nil {
			return nil, fwk.AsStatus(err)
		}
	}

	logger := klog.FromContext(ctx)
	s.namespaceLabels = GetNamespaceLabelsSnapshot(logger, pod.Namespace, pl.nsLister)

	var mustEvaluateHostScopedAntiAffinity bool
	if s.enableInterPodAffinityHostnameFastPath {
		var nodesWithNonHostScopedAntiAffinityPods []fwk.NodeInfo
		var nodesWithAnyRequiredAntiAffinityPods []fwk.NodeInfo
		if nodesWithNonHostScopedAntiAffinityPods, err = pl.sharedLister.NodeInfos().HavePodsWithRequiredNonHostScopedAntiAffinityList(); err != nil {
			return nil, fwk.AsStatus(fmt.Errorf("failed to list NodeInfos with pods with required anti-affinity terms where topologyKey != kubernetes.io/hostname: %w", err))
		}
		if nodesWithAnyRequiredAntiAffinityPods, err = pl.sharedLister.NodeInfos().HavePodsWithRequiredAntiAffinityList(); err != nil {
			return nil, fwk.AsStatus(fmt.Errorf("failed to list NodeInfos with pods with affinity: %w", err))
		}

		s.classifyTermsBasedOnScope()

		var nonHostScopedNodesContainHostScopedTerms bool
		s.existingClusterWideAntiAffinityCounts, nonHostScopedNodesContainHostScopedTerms = pl.getExistingAntiAffinityCounts(ctx, pod, s.namespaceLabels, nodesWithNonHostScopedAntiAffinityPods)
		nodeExistsWithPodsWithOnlyHostScopedAntiAffinityTerms := len(nodesWithAnyRequiredAntiAffinityPods) > len(nodesWithNonHostScopedAntiAffinityPods)
		mustEvaluateHostScopedAntiAffinity = nonHostScopedNodesContainHostScopedTerms || nodeExistsWithPodsWithOnlyHostScopedAntiAffinityTerms

		s.clusterWideAffinityCounts, s.clusterWideAntiAffinityCounts = pl.getIncomingAffinityAntiAffinityCounts(ctx, s.clusterWideAffinityTerms, s.clusterWideAntiAffinityTerms, allNodes)

		// Check if a pod with matching host-scoped affinity exists on the cluster
		if len(s.hostScopedAffinityTerms) > 0 {
			s.matchingHostScopedAffinityPodsCount = pl.countMatchingHostScopedAffinityPodsGlobally(ctx, allNodes, s)
		}
	} else {
		var nodesWithRequiredAntiAffinityPods []fwk.NodeInfo
		if nodesWithRequiredAntiAffinityPods, err = pl.sharedLister.NodeInfos().HavePodsWithRequiredAntiAffinityList(); err != nil {
			return nil, fwk.AsStatus(fmt.Errorf("failed to list NodeInfos with pods with affinity: %w", err))
		}
		s.existingClusterWideAntiAffinityCounts, _ = pl.getExistingAntiAffinityCounts(ctx, pod, s.namespaceLabels, nodesWithRequiredAntiAffinityPods)
		s.clusterWideAffinityCounts, s.clusterWideAntiAffinityCounts = pl.getIncomingAffinityAntiAffinityCounts(ctx, s.podInfo.GetRequiredAffinityTerms(), s.podInfo.GetRequiredAntiAffinityTerms(), allNodes)
		s.clusterWideAffinityTerms = s.podInfo.GetRequiredAffinityTerms()
		s.clusterWideAntiAffinityTerms = s.podInfo.GetRequiredAntiAffinityTerms()
	}

	incomingPodHasNoTerms := len(s.podInfo.GetRequiredAffinityTerms()) == 0 && len(s.podInfo.GetRequiredAntiAffinityTerms()) == 0
	if len(s.existingClusterWideAntiAffinityCounts) == 0 && incomingPodHasNoTerms && !mustEvaluateHostScopedAntiAffinity {
		return nil, fwk.NewStatus(fwk.Skip)
	}

	cycleState.Write(preFilterStateKey, s)
	return nil, nil
}

func (pl *InterPodAffinity) countMatchingHostScopedAffinityPodsGlobally(ctx context.Context, allNodes []fwk.NodeInfo, s *preFilterState) int64 {
	var count int64
	countMatchingPodsOnNode := func(i int) {
		nodeInfo := allNodes[i]
		var localCount int64
		for _, existingPod := range nodeInfo.GetPods() {
			if podMatchesAllAffinityTerms(s.hostScopedAffinityTerms, existingPod.GetPod()) {
				localCount++
			}
		}
		if localCount > 0 {
			atomic.AddInt64(&count, localCount)
		}
	}
	pl.parallelizer.Until(ctx, len(allNodes), countMatchingPodsOnNode, pl.Name())
	return count
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
	if len(state.existingClusterWideAntiAffinityCounts) > 0 {
		// Iterate over topology pairs to get any of the pods being affected by
		// the scheduled pod anti-affinity terms
		for topologyKey, topologyValue := range nodeInfo.Node().Labels {
			tp := topologyPair{key: topologyKey, value: topologyValue}
			if state.existingClusterWideAntiAffinityCounts[tp] > 0 {
				return false
			}
		}
	}

	// Local check for host-scoped anti-affinity
	if state.enableInterPodAffinityHostnameFastPath {
		// Topology label kubernetes.io/hostname must be present on the node for the terms to match.
		if _, ok := nodeInfo.Node().Labels[v1.LabelHostname]; ok {
			for _, existingPodInfo := range nodeInfo.GetPodsWithRequiredAntiAffinity() {
				for _, term := range existingPodInfo.GetRequiredAntiAffinityTerms() {
					if term.TopologyKey == v1.LabelHostname {
						if term.Matches(state.podInfo.GetPod(), state.namespaceLabels) {
							return false
						}
					}
				}
			}
		}
	}

	return true
}

// nodeTopologyConflictsWithAntiAffinity checks if the node's wider topology domain houses pods that conflict with the incoming pod's anti-affinity rules.
// It checks the rules with topology other than kubernetes.io/hostname.
// This is done based on the map of counts (state.clusterWideAntiAffinityCounts) precomputed in PreFilter.
func nodeTopologyConflictsWithAntiAffinity(state *preFilterState, nodeInfo fwk.NodeInfo) bool {
	if len(state.clusterWideAntiAffinityCounts) > 0 {
		for _, term := range state.clusterWideAntiAffinityTerms {
			if topologyValue, ok := nodeInfo.Node().Labels[term.TopologyKey]; ok {
				tp := topologyPair{key: term.TopologyKey, value: topologyValue}
				if state.clusterWideAntiAffinityCounts[tp] > 0 {
					return true
				}
			}
		}
	}
	return false
}

// nodePodsConflictWithAntiAffinity returns true if the target node already hosts pods that conflict with the incoming pod's host-scoped anti-affinity rules.
// It is used for anti-affinity terms with the kubernetes.io/hostname topologyKey.
// Term matching is evaluated dynamically based on the node's PodInfo list because calculating pre-computed counts globally for every individual node would consume excessive resources.
func nodePodsConflictWithAntiAffinity(state *preFilterState, nodeInfo fwk.NodeInfo) bool {
	if len(state.hostScopedAntiAffinityTerms) == 0 {
		return false
	}
	if _, ok := nodeInfo.Node().Labels[v1.LabelHostname]; !ok {
		// Topology label kubernetes.io/hostname must be present on the node for the terms to match.
		return false
	}

	for _, existingPod := range nodeInfo.GetPods() {
		for _, term := range state.hostScopedAntiAffinityTerms {
			if term.Matches(existingPod.GetPod(), nil) {
				return true
			}
		}
	}
	return false
}

// satisfyPodAntiAffinity checks if the incoming pod's anti-affinity rules are satisfied by the existing pods on the node.
// This is used ONLY when the fast-path feature gate is disabled.
func satisfyPodAntiAffinity(state *preFilterState, nodeInfo fwk.NodeInfo) bool {
	if len(state.clusterWideAntiAffinityCounts) > 0 {
		for _, term := range state.podInfo.GetRequiredAntiAffinityTerms() {
			if topologyValue, ok := nodeInfo.Node().Labels[term.TopologyKey]; ok {
				tp := topologyPair{key: term.TopologyKey, value: topologyValue}
				if state.clusterWideAntiAffinityCounts[tp] > 0 {
					return false
				}
			}
		}
	}
	return true
}

// nodePodsSatisfyAffinity performs a fast-path check to see if the target node already hosts at least one pod that satisfies the affinity rules (or if the pod satisfies self-affinity).
// It is used exclusively when ALL affinity terms on the incoming pod have the kubernetes.io/hostname topologyKey, in which case the global check is not required.
// Term matching is evaluated dynamically against the local node's PodInfo list because precomputing counts globally for every individual node would consume excessive resources.
func nodePodsSatisfyAffinity(state *preFilterState, nodeInfo fwk.NodeInfo) bool {
	if len(state.hostScopedAffinityTerms) == 0 {
		return true
	}
	if _, ok := nodeInfo.Node().Labels[v1.LabelHostname]; !ok {
		// Topology label kubernetes.io/hostname must be present on the node for the terms to match.
		return false
	}

	for _, existingPod := range nodeInfo.GetPods() {
		if podMatchesAllAffinityTerms(state.hostScopedAffinityTerms, existingPod.GetPod()) {
			return true
		}
	}
	// existing matching pod not found, check if self-affinity is satisfied
	return satisfiesSelfAffinityCheck(state, nodeInfo)
}

// nodeTopologyMatchesAffinity verifies if the node's topology domain houses at least one pod matching the affinity terms (or if the pod satisfies self-affinity).
// It is used exclusively when the pod has AT LEAST ONE affinity term with a topologyKey other than kubernetes.io/hostname, in which case ALL affinity terms are evaluated here.
// This relies on the map of counts (state.clusterWideAffinityCounts) pre-computed during PreFilter.
func nodeTopologyMatchesAffinity(state *preFilterState, nodeInfo fwk.NodeInfo) bool {
	if len(state.clusterWideAffinityTerms) > 0 {
		podsExist := true
		for _, term := range state.clusterWideAffinityTerms {
			if topologyValue, ok := nodeInfo.Node().Labels[term.TopologyKey]; ok {
				tp := topologyPair{key: term.TopologyKey, value: topologyValue}
				if state.clusterWideAffinityCounts[tp] <= 0 {
					podsExist = false
				}
			} else {
				// All topology labels must exist on the node - also in case of self-affinity.
				return false
			}
		}

		if !podsExist {
			return len(state.clusterWideAffinityCounts) == 0 && satisfiesSelfAffinityCheck(state, nodeInfo)
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
	if state.enableInterPodAffinityHostnameFastPath {
		if state.matchingHostScopedAffinityPodsCount <= 0 && podMatchesAllAffinityTerms(state.podInfo.GetRequiredAffinityTerms(), state.podInfo.GetPod()) {
			for _, term := range state.podInfo.GetRequiredAffinityTerms() {
				if _, ok := nodeInfo.Node().Labels[term.TopologyKey]; !ok {
					return false
				}
			}
			return true
		}
		return false
	}

	// When FG is disabled, topologies were fully evaluated in nodeTopologyMatchesAffinity before this fallback.
	// The `len(state.clusterWideAffinityCounts) == 0` check is performed by the caller.
	if podMatchesAllAffinityTerms(state.podInfo.GetRequiredAffinityTerms(), state.podInfo.GetPod()) {
		return true
	}
	return false
}

// Filter is invoked at the filter extension point to evaluate if a pod can be scheduled on the specified node.
// It uses the maps pre-computed in PreFilter to perform fast O(1) topology checks, and dynamically evaluates
// host-scoped terms locally on the node to avoid traversing the entire cluster.
//
// If the InterPodAffinityHostnameFastPath feature gate is DISABLED:
//  1. Affinity: Uses `clusterWideAffinityCounts` (which contains ALL affinity terms) to verify the node's
//     topology domain houses matching pods. If there are no matching pods globally, and the incoming pod
//     matches its own affinity on this node, it is allowed to schedule.
//  2. Incoming Pod's Anti-Affinity: Uses `clusterWideAntiAffinityCounts` to verify the node's topology
//     domain does not house pods that conflict with the incoming pod's anti-affinity terms.
//  3. Existing Pods' Anti-Affinity: Uses `existingClusterWideAntiAffinityCounts` to verify the node's
//     topology domain does not house existing pods that reject the incoming pod.
//
// If the InterPodAffinityHostnameFastPath feature gate is ENABLED:
// The logic cleanly separates cluster-wide topological checks from host-local pod checks.
//  1. Affinity:
//     - If the incoming pod's affinity terms are EXCLUSIVELY host-scoped, it dynamically iterates over
//     the node's pods to find a match. If there are no matching pods anywhere in the cluster, and the incoming pod
//     matches its own affinity on this node, it is allowed to schedule.
//     - Otherwise, it uses `clusterWideAffinityCounts` to verify topological matches.
//  2. Incoming Pod's Anti-Affinity:
//     - Uses `clusterWideAntiAffinityCounts` to check for conflicts with non-host-scoped terms.
//     - Dynamically iterates over the local node's pods to check for conflicts with host-scoped terms.
//  3. Existing Pods' Anti-Affinity:
//     - Uses `existingClusterWideAntiAffinityCounts` to check for topological rejections from existing pods.
//     - Dynamically iterates over the local node's pods that have host-scoped anti-affinity to see if
//     they reject the incoming pod.
func (pl *InterPodAffinity) Filter(ctx context.Context, cycleState fwk.CycleState, pod *v1.Pod, nodeInfo fwk.NodeInfo) *fwk.Status {
	if pl.enableInPlacePodVerticalScalingSchedulerPreemption && resource.IsPodResizeDeferred(pod) {
		return nil
	}

	state, err := getPreFilterState(cycleState)
	if err != nil {
		return fwk.AsStatus(err)
	}

	if state.enableInterPodAffinityHostnameFastPath {
		if len(state.hostScopedAffinityTerms) > 0 {
			if !nodePodsSatisfyAffinity(state, nodeInfo) {
				return fwk.NewStatus(fwk.UnschedulableAndUnresolvable, ErrReasonAffinityRulesNotMatch)
			}
		} else if len(state.clusterWideAffinityTerms) > 0 {
			if !nodeTopologyMatchesAffinity(state, nodeInfo) {
				return fwk.NewStatus(fwk.UnschedulableAndUnresolvable, ErrReasonAffinityRulesNotMatch)
			}
		}
	} else {
		if !nodeTopologyMatchesAffinity(state, nodeInfo) {
			return fwk.NewStatus(fwk.UnschedulableAndUnresolvable, ErrReasonAffinityRulesNotMatch)
		}
	}

	if state.enableInterPodAffinityHostnameFastPath {
		if nodeTopologyConflictsWithAntiAffinity(state, nodeInfo) || nodePodsConflictWithAntiAffinity(state, nodeInfo) {
			return fwk.NewStatus(fwk.Unschedulable, ErrReasonAntiAffinityRulesNotMatch)
		}
	} else {
		if !satisfyPodAntiAffinity(state, nodeInfo) {
			return fwk.NewStatus(fwk.Unschedulable, ErrReasonAntiAffinityRulesNotMatch)
		}
	}

	if !satisfyExistingPodsAntiAffinity(state, nodeInfo) {
		return fwk.NewStatus(fwk.Unschedulable, ErrReasonExistingAntiAffinityRulesNotMatch)
	}

	return nil
}
