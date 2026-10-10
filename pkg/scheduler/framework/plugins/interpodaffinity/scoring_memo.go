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
	"strconv"
	"strings"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/sets"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodememo"
)

// The scoring half of InterPodAffinity, memoized the way the filtering half is: one contribution per
// node, validated with NodeInfo.GetGeneration, computed from the scheduling cycle snapshot in the
// scheduling goroutine, with only the matching itself spread over the parallelizer. See
// nodememo.NodeMemo for the rules that make a generation enough to decide what is stale, and
// filtering_memo.go for why an entry is keyed by its whole input instead of being paired with an
// invalidation.
//
// What is scored is not reimplemented here: both paths call scoreNode, which is the body PreScore
// always ran, so they cannot drift in what they count - only the traversal is incremental. The tests
// compare them value by value, and node by node through Score, after every mutation of the snapshot.
//
// The aggregate is handed to preScoreState as it is, unlike the filtering counts, which are cloned.
// Score and NormalizeScore only read it, InterPodAffinity has no PreScoreExtensions, and
// preScoreState.Clone shares on purpose, so nothing writes to it behind the memo's back - and with
// topologyKey=kubernetes.io/hostname a scoreMap holds one entry per node, which is precisely the copy
// the memo exists to avoid.

// scoringEntry is the memo of one pod shape's topology scores.
type scoringEntry struct {
	memo *nodememo.NodeMemo[scoreMap]
	// the aggregate, handed to preScoreState as it is. Owned by the memo; everybody else reads it.
	topologyScore scoreMap
	// how many nodes contributed anything at all. PreScore's Skip branch asks whether ANY node did,
	// which is not the same question as "is the aggregate non-empty": two nodes can cancel each other
	// out and leave a topology value at zero, and add drops zeros. Counting the nodes keeps the memo
	// path's Skip decision identical to the full computation's index == -1.
	contributing int64
}

// topologyScoreByMemo is PreScore's cluster walk with the traversal memoized. It returns what
// topologyScoreFull leaves behind: the aggregate, and whether any node contributed.
//
// allNodes is the caller's list, List or HavePodsWithAffinityList depending on hasConstraints. A node
// that stops appearing in it - because it stopped hosting pods with affinity - simply stops being
// passed in, and its contribution leaves the aggregate.
func (pl *InterPodAffinity) topologyScoreByMemo(ctx context.Context, pod *v1.Pod, state *preScoreState,
	allNodes []fwk.NodeInfo, hasConstraints bool) (scoreMap, bool) {

	entry := pl.scoringMemo.GetOrCreate(scoringKey(pod, state, hasConstraints), func() *scoringEntry {
		return &scoringEntry{
			memo:          nodememo.NewNodeMemo[scoreMap](pl.memoParallel()),
			topologyScore: make(scoreMap),
		}
	})

	entry.memo.Refresh(ctx, allNodes,
		func(nodeInfo fwk.NodeInfo) (scoreMap, bool) {
			scores := pl.scoreNode(state, nodeInfo, pod, hasConstraints)
			return scores, len(scores) > 0
		},
		func(old scoreMap, hadOld bool, cur scoreMap, hasCur bool) {
			if hadOld {
				entry.topologyScore.subtract(old)
				entry.contributing--
			}
			if hasCur {
				entry.topologyScore.add(cur)
				entry.contributing++
			}
		})

	return entry.topologyScore, entry.contributing > 0
}

// add folds one node's contribution into the aggregate. It cannot reuse append: append hands the
// inner map over by reference when the topology key is new, which would make the aggregate and the
// stored contribution the same map - and the next add for that key would then mutate a contribution a
// later pass subtracts again, silently doubling it.
//
// A topology value that reaches zero is dropped rather than left at zero, and a topology key left
// without values is dropped with it. That is what keeps the aggregate from growing without bound on a
// cluster whose nodes come and go, since with topologyKey=kubernetes.io/hostname every node that ever
// contributed would otherwise keep an entry forever. It does leave the aggregate differing from the
// full computation's by entries whose value is zero, which cannot change an outcome: Score sums the
// values of the topology keys a node carries, so a zero adds nothing, and NormalizeScore takes its
// min and max over those sums. TestScoringMemoMatchesFullComputation compares both the maps, with
// zeros dropped from both sides, and the score of every node.
func (m scoreMap) add(other scoreMap) {
	for topology, oScores := range other {
		scores := m[topology]
		if scores == nil {
			scores = make(map[string]int64, len(oScores))
			m[topology] = scores
		}
		for value, delta := range oScores {
			scores[value] += delta
			if scores[value] == 0 {
				delete(scores, value)
			}
		}
		if len(scores) == 0 {
			delete(m, topology)
		}
	}
}

// subtract undoes add, so that a recomputed node can replace its previous contribution in the
// aggregate instead of the aggregate being rebuilt from scratch every cycle.
func (m scoreMap) subtract(other scoreMap) {
	for topology, oScores := range other {
		scores := m[topology]
		if scores == nil {
			continue
		}
		for value, delta := range oScores {
			scores[value] -= delta
			if scores[value] == 0 {
				delete(scores, value)
			}
		}
		if len(scores) == 0 {
			delete(m, topology)
		}
	}
}

// scoringKey holds everything a per node contribution depends on that no node generation can express:
// the incoming pod's namespace and labels, because the existing pods' own terms are matched against
// them; the labels of that namespace, because a namespaceSelector term is matched against those and
// relabeling a namespace moves no node generation; the incoming pod's preferred terms with their
// weights; and hasConstraints, which picks both the node list and, per node, GetPods over
// GetPodsWithAffinity.
//
// args.HardPodAffinityWeight is an input too but not part of the key: it is fixed when New builds the
// plugin, and the memo belongs to that same instance, so an instance never mixes two values of it.
// That is the same argument the filtering keys make about the hostname fast path.
func scoringKey(pod *v1.Pod, state *preScoreState, hasConstraints bool) string {
	var b strings.Builder
	b.WriteString(pod.Namespace)
	b.WriteByte('|')
	b.WriteString(labels.Set(pod.Labels).String())
	b.WriteByte('|')
	b.WriteString(namespaceLabelsFingerprint(state.namespaceLabels))
	b.WriteByte('|')
	b.WriteString(strconv.FormatBool(hasConstraints))
	writeWeightedTerms(&b, state.podInfo.GetPreferredAffinityTerms())
	writeWeightedTerms(&b, state.podInfo.GetPreferredAntiAffinityTerms())
	return b.String()
}

// writeWeightedTerms renders weighted terms into a key. termsFingerprint cannot be reused: it takes
// []fwk.AffinityTerm and would drop the weight, and two pod shapes whose preferred terms differ only
// in weight score differently. The terms have had their namespaceSelector merged into Namespaces by
// the time PreScore asks for a key, so Namespaces is what has to be rendered.
func writeWeightedTerms(b *strings.Builder, terms []fwk.WeightedAffinityTerm) {
	b.WriteByte('|')
	for _, t := range terms {
		b.WriteString(t.TopologyKey)
		b.WriteByte(';')
		nodememo.WriteSelector(b, t.Selector)
		b.WriteByte(';')
		nodememo.WriteSelector(b, t.NamespaceSelector)
		b.WriteByte(';')
		b.WriteString(strings.Join(sets.List(t.Namespaces), ","))
		b.WriteByte(';')
		b.WriteString(strconv.FormatInt(int64(t.Weight), 10))
		b.WriteByte(';')
	}
}
