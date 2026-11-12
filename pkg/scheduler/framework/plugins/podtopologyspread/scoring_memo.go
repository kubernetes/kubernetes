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

package podtopologyspread

import (
	"context"
	"fmt"
	"sort"
	"strconv"
	"strings"

	v1 "k8s.io/api/core/v1"
	"k8s.io/component-helpers/scheduling/corev1/nodeaffinity"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodememo"
)

// PreScore's counts, memoized the way InterPodAffinity's are: one contribution per node, validated with
// NodeInfo.GetGeneration, computed from the scheduling cycle snapshot in the scheduling goroutine, with
// only the matching itself spread over the parallelizer. See nodememo.NodeMemo for the rules that make
// a generation enough to decide what is stale, and interpodaffinity's filtering_memo.go for why an
// entry is keyed by its whole input instead of being paired with an invalidation.
//
// What is counted is not reimplemented here: both paths call countNodePods, which is the body PreScore
// always ran. They differ only in which topology values they ask for, and only because they can afford
// to: the full computation asks whether initPreScoreState seeded a value from this cycle's filtered
// nodes, and the memo asks whether the constraint is scoped to anything but the hostname, because a
// memo entry has to mean the same thing in every cycle of a pod shape and the filtered nodes are not
// the same set twice. fill then applies the seeded check the full computation applied, so the state the
// two leave behind is the same.
//
// The memo keeps the aggregate and PreScore copies out of it into the per cycle maps initPreScoreState
// built, rather than sharing it the way InterPodAffinity's scoring memo does. What would be shared is a
// *int64 in a map keyed by this cycle's filtered nodes, which no two cycles agree on, so a copy is what
// the shape asks for anyway - and it is a copy of a few numbers per topology domain, while the walk
// this memo saves is the one over every pod of every node.

// memoParallelMinNodes is how many nodes a pass has to recompute before the matching is spread over the
// parallelizer: below it, waking workers costs more than the matching does. A var rather than a constant
// so that tests can force the parallel path on a fixture of a handful of nodes.
var memoParallelMinNodes = nodememo.DefaultParallelMinNodes

// memoParallel binds this plugin's parallelizer to the memo.
func (pl *PodTopologySpread) memoParallel() nodememo.Parallel {
	return nodememo.Parallel{
		Until: func(ctx context.Context, count int, fn func(index int)) {
			pl.parallelizer.Until(ctx, count, fn, pl.Name())
		},
		MinNodes: memoParallelMinNodes,
	}
}

// spreadCounts is the aggregate over every node: constraint index, then topology value, then the number
// of matching pods. It is the memo's, and PreScore reads it.
type spreadCounts []map[string]int64

func newSpreadCounts(constraints int) spreadCounts {
	counts := make(spreadCounts, constraints)
	for i := range counts {
		counts[i] = make(map[string]int64)
	}
	return counts
}

// add folds one node's contribution into the aggregate.
func (s spreadCounts) add(list nodeSpreadCounts) {
	for _, c := range list {
		if c.index >= len(s) {
			// Cannot happen: the aggregate is sized from the same constraints the contributions were
			// computed with, and a change of constraints is a change of key. Skipping the entry keeps a
			// mismatch from panicking in the scheduling goroutine.
			continue
		}
		s[c.index][c.value] += int64(c.count)
	}
}

// subtract undoes add, so that a recomputed node can replace its previous contribution in the aggregate
// instead of the aggregate being rebuilt from scratch every cycle. A topology value that reaches zero
// is dropped rather than left at zero: with a hostname scoped topology key a domain is a node, and a
// cluster whose nodes come and go would otherwise grow the map without bound. fill only ever reads a
// value it finds, so a dropped zero and a stored one fill the same slot with the same number.
func (s spreadCounts) subtract(list nodeSpreadCounts) {
	for _, c := range list {
		if c.index >= len(s) {
			continue
		}
		values := s[c.index]
		values[c.value] -= int64(c.count)
		if values[c.value] <= 0 {
			delete(values, c.value)
		}
	}
}

// fill copies the aggregate into the per cycle counts of a preScoreState. Only the topology values
// initPreScoreState seeded are filled, which is the check the full computation makes before it counts
// and the memo makes after: a node whose domain no filtered node is in contributes to the aggregate and
// is then not copied, and a hostname scoped constraint contributes nothing at all because countNodePods
// never asks for one.
func (s spreadCounts) fill(target []map[string]*int64) {
	for i, values := range s {
		if i >= len(target) {
			continue
		}
		for value, count := range values {
			if slot := target[i][value]; slot != nil {
				*slot += count
			}
		}
	}
}

// spreadEntry is the memo of one pod shape.
type spreadEntry struct {
	memo *nodememo.NodeMemo[nodeSpreadCounts]
	// the aggregate. Owned by the memo; PreScore copies out of it.
	counts spreadCounts
}

// addTopologyCountsByMemo fills state.TopologyValueToPodCounts with what addTopologyCountsFull would
// put there, reusing the contribution of every node whose generation did not move since this pod shape
// was last scored. nodes is the snapshot's whole node list, the same one the full computation walks.
func (pl *PodTopologySpread) addTopologyCountsByMemo(ctx context.Context, pod *v1.Pod,
	state *preScoreState, nodes []fwk.NodeInfo, requireAllTopologies bool,
	requiredNodeAffinity nodeaffinity.RequiredNodeAffinity) {

	constraints := state.Constraints
	entry := pl.scoringMemo.GetOrCreate(spreadKey(pod, constraints, requireAllTopologies),
		func() *spreadEntry {
			return &spreadEntry{
				memo:   nodememo.NewNodeMemo[nodeSpreadCounts](pl.memoParallel()),
				counts: newSpreadCounts(len(constraints)),
			}
		})

	in := &spreadInputs{
		logger:               klog.FromContext(ctx),
		pod:                  pod,
		constraints:          constraints,
		requireAllTopologies: requireAllTopologies,
		requiredNodeAffinity: requiredNodeAffinity,
		wanted: func(i int, _ string) bool {
			return constraints[i].TopologyKey != v1.LabelHostname
		},
	}

	entry.memo.Refresh(ctx, nodes,
		func(nodeInfo fwk.NodeInfo) (nodeSpreadCounts, bool) {
			counts := pl.countNodePods(nodeInfo, in)
			return counts, len(counts) > 0
		},
		func(old nodeSpreadCounts, hadOld bool, cur nodeSpreadCounts, hasCur bool) {
			if hadOld {
				entry.counts.subtract(old)
			}
			if hasCur {
				entry.counts.add(cur)
			}
		})

	entry.counts.fill(state.TopologyValueToPodCounts)
}

// spreadKey holds everything a per node contribution depends on that no node generation can express:
// the incoming pod's namespace, because countPodsMatchSelector counts only pods in it; the constraints,
// which are what the counts are of; requireAllTopologies, which decides whether a node missing a
// topology key contributes at all; and the pod's node selector, required node affinity and tolerations,
// which the two node level checks and a constraint with an inclusion policy of Honor are matched
// against per node.
//
// The incoming pod's labels are deliberately not part of it. Nothing in the per node computation looks
// at them - the constraints were derived from them by initPreScoreState and are in here themselves - so
// two workloads whose pods spread the same way share one entry, which is what makes the memo worth its
// bookkeeping on a cluster that schedules more than one.
//
// Neither are the plugin's feature gates: they are fixed when New builds the plugin, and the memo
// belongs to that same instance.
func spreadKey(pod *v1.Pod, constraints []topologySpreadConstraint, requireAllTopologies bool) string {
	var b strings.Builder
	b.WriteString(pod.Namespace)
	b.WriteByte('|')
	b.WriteString(strconv.FormatBool(requireAllTopologies))
	for _, c := range constraints {
		b.WriteByte('|')
		b.WriteString(c.TopologyKey)
		b.WriteByte(';')
		nodememo.WriteSelector(&b, c.Selector)
		// MaxSkew and MinDomains are what Score and NormalizeScore read, not what a node contributes,
		// but two constraints that differ in them are different constraints to the caller, and indexing
		// the aggregate by a shared key would hand one constraint's counts to the other.
		fmt.Fprintf(&b, ";%d;%d;%s;%s", c.MaxSkew, c.MinDomains, c.NodeAffinityPolicy, c.NodeTaintsPolicy)
	}
	b.WriteByte('|')
	writeNodeSelectorFingerprint(&b, pod)
	writeTolerationsFingerprint(&b, pod.Spec.Tolerations)
	return b.String()
}

// writeNodeSelectorFingerprint renders what nodeaffinity.GetRequiredNodeAffinity matches with: the pod's
// nodeSelector and its required node affinity. Both are consulted per node - once for the whole node
// when the inclusion policies are off, and per constraint when NodeAffinityPolicy is Honor - and both
// come from the incoming pod's spec.
func writeNodeSelectorFingerprint(b *strings.Builder, pod *v1.Pod) {
	// Sorted, because a map's iteration order is not: the same nodeSelector has to render to the same
	// string every cycle, or the entry would be a new one each time.
	keys := make([]string, 0, len(pod.Spec.NodeSelector))
	for key := range pod.Spec.NodeSelector {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		fmt.Fprintf(b, "%s=%s;", key, pod.Spec.NodeSelector[key])
	}
	affinity := pod.Spec.Affinity
	if affinity == nil || affinity.NodeAffinity == nil ||
		affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution == nil {
		return
	}
	for _, term := range affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms {
		// One bracketed group per term, because a term is a group: the requirements inside one are ANDed
		// and the terms themselves are ORed, and the two shapes admit different nodes - those matching
		// every requirement, versus those matching the requirements of any one term. The boundary
		// separates the terms and the tag writeNodeSelectorRequirements puts in front of each of its two
		// requirement lists separates those. That is redundant for the shapes a pod can actually carry and
		// it is deliberate: a rendering that is ambiguous for some pair of shapes hands one pod shape
		// another's counts, and neither delimiter then has to argue that no key, value or operator can
		// contain the other's.
		b.WriteByte('{')
		writeNodeSelectorRequirements(b, "E", term.MatchExpressions)
		writeNodeSelectorRequirements(b, "F", term.MatchFields)
		b.WriteByte('}')
	}
}

// writeNodeSelectorRequirements renders one of a term's two requirement lists behind tag, which records
// whether they came from MatchExpressions or from MatchFields: the same requirement means a different
// thing in each, since one is matched against the node's labels and the other against its fields. The
// ';' in front of the tag keeps a value that happens to end in the tag's letter from swallowing it, and
// an empty list is rendered as the tag on its own rather than as nothing, so that a term with one
// requirement in each list cannot render as a term with both in the first.
func writeNodeSelectorRequirements(b *strings.Builder, tag string, requirements []v1.NodeSelectorRequirement) {
	b.WriteByte(';')
	b.WriteString(tag)
	for _, r := range requirements {
		fmt.Fprintf(b, "|%s;%s;%s", r.Key, r.Operator, strings.Join(r.Values, ","))
	}
}

// writeTolerationsFingerprint renders the pod's tolerations, which a constraint with
// NodeTaintsPolicy=Honor matches a node's taints against.
func writeTolerationsFingerprint(b *strings.Builder, tolerations []v1.Toleration) {
	for _, t := range tolerations {
		seconds := int64(-1)
		if t.TolerationSeconds != nil {
			seconds = *t.TolerationSeconds
		}
		fmt.Fprintf(b, "|%s;%s;%s;%s;%d", t.Key, t.Operator, t.Value, t.Effect, seconds)
	}
}
