/*
Copyright The Kubernetes Authors.

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
	"sort"
	"strings"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/sets"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodememo"
)

// The filtering counts of InterPodAffinity, memoized across scheduling cycles and validated with
// NodeInfo.GetGeneration instead of being recomputed from scratch every cycle. See nodememo.NodeMemo
// for why that shape, and in particular for why the counts are derived from the scheduling cycle
// snapshot rather than maintained from informer events.
//
// Nothing here reimplements the matching. countExistingAntiAffinityOnNode, countIncomingOnNode and
// countMatchingHostScopedAffinityPodsOnNode are the per node bodies the full computation in
// filtering.go runs too, so the two paths cannot drift in what they count - only the traversal is
// incremental. The tests compare them value by value after every mutation of the snapshot.
//
// A memo entry is kept per pod shape, keyed by everything a per node contribution depends on that no
// node generation can express. A key coarser than the inputs would have to be paired with an
// invalidation, and every invalidation costs a cold pass, which is dearer than the full computation
// the memo replaces - so two shapes alternating would leave the memo slower than having none.

// memoParallelMinNodes is how many nodes a pass has to recompute before the matching is spread over
// the parallelizer: below it, waking workers costs more than the matching does. A var rather than a
// constant so that tests can force the parallel path on a fixture of a handful of nodes.
var memoParallelMinNodes = nodememo.DefaultParallelMinNodes

// memoParallel binds this plugin's parallelizer to the memo.
func (pl *InterPodAffinity) memoParallel() nodememo.Parallel {
	return nodememo.Parallel{
		Until: func(ctx context.Context, count int, fn func(index int)) {
			pl.parallelizer.Until(ctx, count, fn, pl.Name())
		},
		MinNodes: memoParallelMinNodes,
	}
}

// existingNodeCounts is what one node contributes to the existing pods' anti-affinity counts.
type existingNodeCounts struct {
	counts topologyToMatchedTermCountList
	// hostScopedTerms is how many hostname scoped required anti-affinity terms the node's pods carry.
	// A count rather than the bool the full computation reduces it to, because the memo folds
	// contributions into an aggregate and takes them back out again, and an OR cannot be undone: a node
	// that stops carrying such a term has to be able to withdraw its contribution. "Is the sum
	// positive" asks the same question as "did any node have one".
	hostScopedTerms int64
}

// incomingNodeCounts is what one node contributes to the incoming pod's own (anti-)affinity counts.
type incomingNodeCounts struct {
	affinity     topologyToMatchedTermCountList
	antiAffinity topologyToMatchedTermCountList
}

// existingCountsEntry is the memo of one pod shape of the existing pods' anti-affinity counts: the
// required anti-affinity terms of the pods already on a node, matched against the incoming pod.
type existingCountsEntry struct {
	memo *nodememo.NodeMemo[existingNodeCounts]
	// the aggregate. Owned by the memo; callers get a clone of it, because preFilterState hands its
	// maps to updateWithPod and a mutation must never reach an aggregate later cycles keep reading.
	counts          topologyToMatchedTermCount
	hostScopedTerms int64
}

// incomingCountsEntry is the memo of one set of the incoming pod's required terms: the pods on a node
// matched against those terms.
type incomingCountsEntry struct {
	memo           *nodememo.NodeMemo[incomingNodeCounts]
	affinityCounts topologyToMatchedTermCount
	antiAffinity   topologyToMatchedTermCount
}

// hostScopedAffinityEntry is the memo of one set of host scoped affinity terms: how many pods in the
// cluster match all of them.
type hostScopedAffinityEntry struct {
	memo  *nodememo.NodeMemo[int64]
	count int64
}

// existingAntiAffinityCounts is getExistingAntiAffinityCounts with the walk memoized, and the single
// place that chooses between the two. useMemo is false in a pod group scheduling cycle: that cycle
// assumes every pod of the gang into one snapshot in place, Snapshot.AssumePod restores the old
// generation on purpose to stay consistent with the cache, and Cache.UpdateSnapshot runs once for the
// whole group - a generation that does not move while the node does is the one thing the memo cannot
// detect. Counting from the snapshot itself is also what lets the later pods of a gang see the earlier
// ones.
func (pl *InterPodAffinity) existingAntiAffinityCounts(ctx context.Context, incomingPod *v1.Pod,
	nsLabels labels.Set, nodes []fwk.NodeInfo, useMemo bool) (topologyToMatchedTermCount, bool) {
	if !useMemo {
		return pl.getExistingAntiAffinityCounts(ctx, incomingPod, nsLabels, nodes)
	}

	entry := pl.filteringExistingMemo.GetOrCreate(existingCountsKey(incomingPod, nsLabels), func() *existingCountsEntry {
		return &existingCountsEntry{
			memo:   nodememo.NewNodeMemo[existingNodeCounts](pl.memoParallel()),
			counts: topologyToMatchedTermCount{},
		}
	})

	entry.memo.Refresh(ctx, nodes,
		func(nodeInfo fwk.NodeInfo) (existingNodeCounts, bool) {
			counts := pl.countExistingAntiAffinityOnNode(nodeInfo, incomingPod, nsLabels)
			// A node that carries only hostname scoped terms still has something to say, so it has to be
			// stored: leaving it out would let the next pass read it as "contributes nothing".
			return counts, len(counts.counts) > 0 || counts.hostScopedTerms > 0
		},
		func(old existingNodeCounts, hadOld bool, cur existingNodeCounts, hasCur bool) {
			if hadOld {
				entry.counts.subtractWithList(old.counts)
				entry.hostScopedTerms -= old.hostScopedTerms
			}
			if hasCur {
				entry.counts.mergeWithList(cur.counts)
				entry.hostScopedTerms += cur.hostScopedTerms
			}
		})

	// A clone, not the aggregate itself: preFilterState's maps are mutated by updateWithPod, which
	// preemption and the nominated pods path reach through AddPod and RemovePod. They do it on a clone
	// of the cycle state today, and preFilterState.Clone deep copies these very maps, so the aggregate
	// would survive - but that is an invariant about other code, and the copy is cheap next to the walk
	// it replaces.
	return entry.counts.clone(), entry.hostScopedTerms > 0
}

// incomingAffinityAntiAffinityCounts is getIncomingAffinityAntiAffinityCounts with the walk memoized.
func (pl *InterPodAffinity) incomingAffinityAntiAffinityCounts(ctx context.Context, affinityTerms,
	antiAffinityTerms []fwk.AffinityTerm, allNodes []fwk.NodeInfo, useMemo bool) (topologyToMatchedTermCount, topologyToMatchedTermCount) {
	if !useMemo {
		return pl.getIncomingAffinityAntiAffinityCounts(ctx, affinityTerms, antiAffinityTerms, allNodes)
	}
	if len(affinityTerms) == 0 && len(antiAffinityTerms) == 0 {
		// Nothing to match, so nothing to memoize - getIncomingAffinityAntiAffinityCounts returns here
		// too, and a memo pass would walk every node to store nothing and keep an entry for a key whose
		// aggregate is empty by construction. PreFilter reaches this whenever the pod has no terms of its
		// own but something else keeps it out of the Skip branch.
		return topologyToMatchedTermCount{}, topologyToMatchedTermCount{}
	}

	entry := pl.filteringIncomingMemo.GetOrCreate(incomingCountsKey(affinityTerms, antiAffinityTerms), func() *incomingCountsEntry {
		return &incomingCountsEntry{
			memo:           nodememo.NewNodeMemo[incomingNodeCounts](pl.memoParallel()),
			affinityCounts: topologyToMatchedTermCount{},
			antiAffinity:   topologyToMatchedTermCount{},
		}
	})

	entry.memo.Refresh(ctx, allNodes,
		func(nodeInfo fwk.NodeInfo) (incomingNodeCounts, bool) {
			counts := countIncomingOnNode(nodeInfo, affinityTerms, antiAffinityTerms)
			return counts, len(counts.affinity) > 0 || len(counts.antiAffinity) > 0
		},
		func(old incomingNodeCounts, hadOld bool, cur incomingNodeCounts, hasCur bool) {
			if hadOld {
				entry.affinityCounts.subtractWithList(old.affinity)
				entry.antiAffinity.subtractWithList(old.antiAffinity)
			}
			if hasCur {
				entry.affinityCounts.mergeWithList(cur.affinity)
				entry.antiAffinity.mergeWithList(cur.antiAffinity)
			}
		})

	return entry.affinityCounts.clone(), entry.antiAffinity.clone()
}

// matchingHostScopedAffinityPodsCount is countMatchingHostScopedAffinityPodsGlobally with the walk
// memoized.
func (pl *InterPodAffinity) matchingHostScopedAffinityPodsCount(ctx context.Context,
	terms []fwk.AffinityTerm, allNodes []fwk.NodeInfo, useMemo bool) int64 {
	if !useMemo {
		return pl.countMatchingHostScopedAffinityPodsGlobally(ctx, allNodes, terms)
	}

	entry := pl.filteringHostScopedAffinityMemo.GetOrCreate(incomingCountsKey(terms, nil), func() *hostScopedAffinityEntry {
		return &hostScopedAffinityEntry{memo: nodememo.NewNodeMemo[int64](pl.memoParallel())}
	})

	entry.memo.Refresh(ctx, allNodes,
		func(nodeInfo fwk.NodeInfo) (int64, bool) {
			count := countMatchingHostScopedAffinityPodsOnNode(nodeInfo, terms)
			return count, count > 0
		},
		func(old int64, hadOld bool, cur int64, hasCur bool) {
			if hadOld {
				entry.count -= old
			}
			if hasCur {
				entry.count += cur
			}
		})

	return entry.count
}

// existingCountsKey keys the existing pods' counts. Their contributions match the terms of the pods
// already on a node against the incoming pod, so the incoming pod's namespace and labels are part of
// the key - and so are the labels of its namespace, because a namespaceSelector term is matched
// against them and relabeling a namespace moves no node generation.
//
// Which nodes are walked is not in the key: the caller passes either
// HavePodsWithRequiredNonHostScopedAntiAffinityList or HavePodsWithRequiredAntiAffinityList depending
// on the InterPodAffinityHostnameFastPath gate, and both the gate and the memo belong to one plugin
// instance, which one profile owns, so an instance never mixes the two lists.
func existingCountsKey(pod *v1.Pod, nsLabels labels.Set) string {
	return pod.Namespace + "|" + labels.Set(pod.Labels).String() + "|" + namespaceLabelsFingerprint(nsLabels)
}

// incomingCountsKey keys a set of the incoming pod's own required terms. The terms are the whole
// input to the per node computation.
func incomingCountsKey(termsList ...[]fwk.AffinityTerm) string {
	return termsFingerprint(termsList...)
}

// namespaceLabelsFingerprint turns the namespace labels a match depends on into something cheap to
// compare. A namespace relabel changes what a namespaceSelector matches and bumps no node generation,
// so without this the memo would keep serving contributions computed against the old labels.
// encoding/json is avoided on purpose: it sorts map keys, which is what makes this stable.
func namespaceLabelsFingerprint(nsLabels labels.Set) string {
	if len(nsLabels) == 0 {
		return ""
	}
	keys := make([]string, 0, len(nsLabels))
	for key := range nsLabels {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	var b strings.Builder
	for _, key := range keys {
		fmt.Fprintf(&b, "%s=%s;", key, nsLabels[key])
	}
	return b.String()
}

// termsFingerprint renders a list of terms into a key. It cannot marshal them: AffinityTerm carries
// labels.Selector, an interface whose concrete types are made of unexported fields, so JSON would
// render every selector as {} and make different terms look identical. Namespaces is sorted by
// sets.List, and selectors go through nodememo.WriteSelector.
func termsFingerprint(termsList ...[]fwk.AffinityTerm) string {
	var b strings.Builder
	for _, terms := range termsList {
		b.WriteByte('|')
		for _, t := range terms {
			b.WriteString(t.TopologyKey)
			b.WriteByte(';')
			nodememo.WriteSelector(&b, t.Selector)
			b.WriteByte(';')
			nodememo.WriteSelector(&b, t.NamespaceSelector)
			b.WriteByte(';')
			b.WriteString(strings.Join(sets.List(t.Namespaces), ","))
			b.WriteByte(';')
		}
	}
	return b.String()
}
