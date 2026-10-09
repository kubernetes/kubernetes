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
	"testing"

	"github.com/google/go-cmp/cmp"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/backend/cache"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodememo"
	plugintesting "k8s.io/kubernetes/pkg/scheduler/framework/plugins/testing"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

// spreadResult is what a PreScore run leaves behind, reduced to what the memo produces and to what the
// framework goes on to do with it. Comparing the scores is the assertion that matters; comparing the
// counts is the one that localizes a failure.
type spreadResult struct {
	Skipped bool
	// Counts is preScoreState.TopologyValueToPodCounts with its pointers dereferenced, so that cmp can
	// diff it without being told about them.
	Counts     []map[string]int64
	Weights    []float64
	Ignored    []string
	Scores     map[string]int64
	Normalized map[string]int64
}

// newSpreadMemoPlugin returns two plugins over one snapshot: the one New builds, which memoizes, and a
// copy of it with the memo taken out, which therefore takes the full computation. Everything else about
// the two is identical, so a difference between them is a difference the memo made.
func newSpreadMemoPlugin(t *testing.T, ctx context.Context, snapshot *cache.Snapshot, fts feature.Features,
	objs []runtime.Object) (memo, full *PodTopologySpread) {
	t.Helper()
	p := plugintesting.SetupPluginWithInformers(ctx, t, frameworkruntime.FactoryAdapter(fts, New),
		&config.PodTopologySpreadArgs{DefaultingType: config.ListDefaulting}, snapshot, objs)
	memoPlugin := p.(*PodTopologySpread)
	if !memoPlugin.scoringMemoReady() {
		t.Fatalf("New did not build the scoring memo")
	}
	fullPlugin := *memoPlugin
	fullPlugin.scoringMemo = nil
	if fullPlugin.scoringMemoReady() {
		t.Fatalf("the plugin meant to take the full computation still has a memo")
	}
	return memoPlugin, &fullPlugin
}

// runPreScoreSpread drives one plugin through PreScore, Score and NormalizeScore over every node of the
// snapshot. filteredNodes is named rather than taken as node infos, because which nodes a caller passes
// as filtered is half of what these tests are about: initPreScoreState seeds the per cycle maps from
// them, and a topology value that is not seeded is one the full computation never counts and the memo
// counts and then drops.
func runPreScoreSpread(t *testing.T, ctx context.Context, pl *PodTopologySpread, pod *v1.Pod,
	snapshot *cache.Snapshot, filteredNodeNames ...string) spreadResult {
	t.Helper()
	if _, err := snapshot.NodeInfos().List(); err != nil {
		t.Fatalf("List: %v", err)
	}
	filtered := make([]fwk.NodeInfo, 0, len(filteredNodeNames))
	for _, name := range filteredNodeNames {
		nodeInfo, err := snapshot.NodeInfos().Get(name)
		if err != nil {
			t.Fatalf("Get(%s): %v", name, err)
		}
		filtered = append(filtered, nodeInfo)
	}

	state := framework.NewCycleState()
	status := pl.PreScore(ctx, state, pod, filtered)
	if status != nil && status.Code() == fwk.Skip {
		return spreadResult{Skipped: true}
	}
	if !status.IsSuccess() {
		t.Fatalf("PreScore(%s): %v", pod.Name, status)
	}
	s, err := getPreScoreState(state)
	if err != nil {
		t.Fatalf("getPreScoreState(%s): %v", pod.Name, err)
	}

	result := spreadResult{
		Counts:     make([]map[string]int64, len(s.TopologyValueToPodCounts)),
		Weights:    s.TopologyNormalizingWeight,
		Ignored:    sets.List(s.IgnoredNodes),
		Scores:     make(map[string]int64, len(filtered)),
		Normalized: make(map[string]int64, len(filtered)),
	}
	for i, byValue := range s.TopologyValueToPodCounts {
		dereferenced := make(map[string]int64, len(byValue))
		for value, count := range byValue {
			dereferenced[value] = *count
		}
		result.Counts[i] = dereferenced
	}
	// Score is run over the filtered nodes, which is what the framework does: it scores the nodes that
	// passed Filter, and those are the ones initPreScoreState seeded a topology value for or put in
	// IgnoredNodes. Scoring a node outside that set would dereference a slot Score assumes is there.
	scoreList := make(fwk.NodeScoreList, 0, len(filtered))
	for _, nodeInfo := range filtered {
		name := nodeInfo.Node().Name
		score, status := pl.Score(ctx, state, pod, nodeInfo)
		if status != nil && !status.IsSuccess() {
			t.Fatalf("Score(%s/%s): %v", pod.Name, name, status)
		}
		result.Scores[name] = score
		scoreList = append(scoreList, fwk.NodeScore{Name: name, Score: score})
	}
	if status := pl.NormalizeScore(ctx, state, pod, scoreList); status != nil && !status.IsSuccess() {
		t.Fatalf("NormalizeScore(%s): %v", pod.Name, status)
	}
	for _, nodeScore := range scoreList {
		result.Normalized[nodeScore.Name] = nodeScore.Score
	}
	return result
}

// spreadMemoFixture builds five nodes and the pods that make every branch of countNodePods matter:
//   - node1 and node2 are in zone z1, node3 and node4 in zone z2, so a zone count is a sum over more
//     than one node and a node that moves between zones moves a count;
//   - node4 is tainted, which is what a constraint with NodeTaintsPolicy=Honor excludes;
//   - node5 has no zone label, so requireAllTopologies puts it in IgnoredNodes and countNodePods never
//     reaches its pods;
//   - every node hosts one matching pod, and ns2 hosts one with the same labels, which
//     countPodsMatchSelector must not count.
func spreadMemoFixture() *cache.Snapshot {
	nodes := []*v1.Node{
		st.MakeNode().Name("node1").Label("zone", "z1").Label(v1.LabelHostname, "node1").Obj(),
		st.MakeNode().Name("node2").Label("zone", "z1").Label(v1.LabelHostname, "node2").Obj(),
		st.MakeNode().Name("node3").Label("zone", "z2").Label(v1.LabelHostname, "node3").Obj(),
		st.MakeNode().Name("node4").Label("zone", "z2").Label(v1.LabelHostname, "node4").
			Taints([]v1.Taint{{Key: "dedicated", Value: "other", Effect: v1.TaintEffectNoSchedule}}).Obj(),
		st.MakeNode().Name("node5").Label(v1.LabelHostname, "node5").Obj(),
		// A node in a zone of its own that hosts nothing the constraints select: it contributes nothing,
		// and a memo that stored a zero contribution for it would look correct while defeating the rule
		// that lets a later pass skip a node it already knows to be empty.
		st.MakeNode().Name("node6").Label("zone", "z3").Label(v1.LabelHostname, "node6").Obj(),
	}
	web := func(name, node string) *v1.Pod {
		return st.MakePod().Namespace("ns1").Name(name).UID(name).Label("app", "web").Node(node).Obj()
	}
	pods := []*v1.Pod{
		web("web-a", "node1"),
		web("web-b", "node2"),
		web("web-c", "node3"),
		web("web-d", "node4"),
		web("web-e", "node5"),
		st.MakePod().Namespace("ns2").Name("web-other").UID("web-other").Label("app", "web").Node("node1").Obj(),
	}
	return cache.NewSnapshot(pods, nodes)
}

// spreadMemoAllNodes is every node of spreadMemoFixture, in snapshot order.
var spreadMemoAllNodes = []string{"node1", "node2", "node3", "node4", "node5", "node6"}

var spreadMemoNamespaces = []runtime.Object{
	&v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: "ns1", Labels: map[string]string{"team": "infra"}}},
	&v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: "ns2", Labels: map[string]string{"team": "infra"}}},
}

// spreadMemoIncoming is a pod with three soft constraints over the same selector: one zone scoped, one
// hostname scoped - which initPreScoreState never seeds, because Score counts a node's own pods itself,
// and which is therefore the constraint countNodePods has to leave alone - and one zone scoped with
// NodeTaintsPolicy=Honor, which only differs from the first while
// EnableNodeInclusionPolicyInPodTopologySpread is on.
func spreadMemoIncoming(namespace, name string, mutate func(*v1.Pod)) *v1.Pod {
	selector := &metav1.LabelSelector{MatchLabels: map[string]string{"app": "web"}}
	honor := v1.NodeInclusionPolicyHonor
	pod := st.MakePod().Namespace(namespace).Name(name).UID(name).Label("app", "web").
		SpreadConstraint(1, "zone", v1.ScheduleAnyway, selector, nil, nil, nil, nil).
		SpreadConstraint(1, v1.LabelHostname, v1.ScheduleAnyway, selector, nil, nil, nil, nil).
		SpreadConstraint(2, "zone", v1.ScheduleAnyway, selector, nil, nil, &honor, nil).Obj()
	if mutate != nil {
		mutate(pod)
	}
	return pod
}

func TestSpreadMemoMatchesFullComputation(t *testing.T) {
	spreadMemoMatchesFullComputation(t, false, nodememo.DefaultParallelMinNodes)
}

// TestSpreadMemoMatchesFullComputationInParallel runs the same steps with the parallel threshold at
// zero, so every pass spreads the matching over the parallelizer and has to land on the same numbers.
func TestSpreadMemoMatchesFullComputationInParallel(t *testing.T) {
	spreadMemoMatchesFullComputation(t, false, 0)
}

// TestSpreadMemoMatchesFullComputationWithInclusionPolicies runs the same steps with the gate on, which
// is what makes the third constraint honor node taints and matchNodeInclusionPolicies part of the walk.
func TestSpreadMemoMatchesFullComputationWithInclusionPolicies(t *testing.T) {
	spreadMemoMatchesFullComputation(t, true, nodememo.DefaultParallelMinNodes)
}

func spreadMemoMatchesFullComputation(t *testing.T, inclusionPolicy bool, parallelMinNodes int) {
	previous := memoParallelMinNodes
	memoParallelMinNodes = parallelMinNodes
	defer func() { memoParallelMinNodes = previous }()

	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	snapshot := spreadMemoFixture()
	fts := feature.Features{EnableNodeInclusionPolicyInPodTopologySpread: inclusionPolicy}
	memoPlugin, fullPlugin := newSpreadMemoPlugin(t, ctx, snapshot, fts, spreadMemoNamespaces)

	all := spreadMemoAllNodes
	incoming := spreadMemoIncoming("ns1", "incoming", nil)
	// A pod that may only land in z1: the node level required affinity check excludes every other node
	// from the walk, and it is a different memo key, so it is a different entry over the same nodes.
	restricted := spreadMemoIncoming("ns1", "restricted", func(pod *v1.Pod) {
		pod.Spec.NodeSelector = map[string]string{"zone": "z1"}
	})

	nodeInfoOf := func(name string) fwk.NodeInfo {
		t.Helper()
		nodeInfo, err := snapshot.NodeInfos().Get(name)
		if err != nil {
			t.Fatalf("Get(%s): %v", name, err)
		}
		return nodeInfo
	}
	addPod := func(pod *v1.Pod) {
		t.Helper()
		podInfo, err := framework.NewPodInfo(pod)
		if err != nil {
			t.Fatalf("NewPodInfo(%s): %v", pod.Name, err)
		}
		nodeInfoOf(pod.Spec.NodeName).AddPodInfo(podInfo)
	}
	removePod := func(pod *v1.Pod) {
		t.Helper()
		if err := nodeInfoOf(pod.Spec.NodeName).RemovePod(klog.FromContext(ctx), pod); err != nil {
			t.Fatalf("RemovePod(%s): %v", pod.Name, err)
		}
	}
	relabelNode := func(name, zone string) {
		t.Helper()
		node := nodeInfoOf(name).Node().DeepCopy()
		if zone == "" {
			delete(node.Labels, "zone")
		} else {
			node.Labels["zone"] = zone
		}
		nodeInfoOf(name).SetNode(node)
	}

	step := func(name string, pod *v1.Pod, filtered ...string) {
		t.Helper()
		want := runPreScoreSpread(t, ctx, fullPlugin, pod, snapshot, filtered...)
		got := runPreScoreSpread(t, ctx, memoPlugin, pod, snapshot, filtered...)
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("%s: the memo's counts differ from the full computation (-want,+got):\n%s", name, diff)
		}
		if want.Skipped {
			t.Fatalf("%s: PreScore Skipped, so the step proves nothing about the memo", name)
		}
		if len(want.Counts) == 0 || want.Counts[0]["z1"] == 0 {
			t.Fatalf("%s: the fixture produced no zone counts at all, so the step proves nothing", name)
		}
	}

	step("initial", incoming, all...)
	// Only z1 is seeded, so every count for z2 is computed and then dropped: this is the step that
	// separates the two callers' wanted, and a memo that filled a slot initPreScoreState never made
	// would panic here rather than differ.
	step("only z1 among the filtered nodes", incoming, "node1", "node2")
	step("the restricted pod", restricted, all...)

	added := st.MakePod().Namespace("ns1").Name("web-added").UID("web-added").
		Label("app", "web").Node("node3").Obj()
	addPod(added)
	step("pod added", incoming, all...)

	// ... and removed again, which is the half that needs subtract: a memo that could only add would
	// keep counting it.
	removePod(added)
	step("pod removed", incoming, all...)

	// A terminating pod must not be counted, by either path.
	terminating := st.MakePod().Namespace("ns1").Name("web-terminating").UID("web-terminating").
		Label("app", "web").Node("node1").Terminating().Obj()
	addPod(terminating)
	step("terminating pod added", incoming, all...)
	removePod(terminating)

	// node3 moves to a zone of its own, which moves a count out of z2, and back again.
	relabelNode("node3", "z3")
	step("node relabelled", incoming, all...)
	relabelNode("node3", "z2")
	step("node relabelled back", incoming, all...)

	// A node loses its zone label altogether: it stops contributing, and with requireAllTopologies it
	// joins IgnoredNodes as well.
	relabelNode("node4", "")
	step("node lost its zone label", incoming, all...)
	relabelNode("node4", "z2")
	step("node got its zone label back", incoming, all...)

	// A pod that matches nothing the constraints select: no node's contribution changes, and the memo
	// has to say so rather than store a zero for it.
	plain := st.MakePod().Namespace("ns1").Name("plain").UID("plain").Label("app", "other").Node("node2").Obj()
	addPod(plain)
	step("pod matching no constraint added", incoming, all...)
	removePod(plain)
}

// onlySpreadEntry returns the single entry a test's pod shape is stored under.
func onlySpreadEntry(t *testing.T, pl *PodTopologySpread) *spreadEntry {
	t.Helper()
	keys := pl.scoringMemo.Keys()
	if len(keys) != 1 {
		t.Fatalf("the scoring memo holds %d entries, want exactly one: %v", len(keys), keys)
	}
	entry, ok := pl.scoringMemo.Get(keys[0])
	if !ok {
		t.Fatalf("the scoring memo lost the entry for %q between listing its keys and reading it", keys[0])
	}
	return entry
}

// TestSpreadMemoReusesUnchangedNodes is the other half: correctness alone would also be served by
// recomputing every node every cycle, which is what the memo exists to avoid.
func TestSpreadMemoReusesUnchangedNodes(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	snapshot := spreadMemoFixture()
	memoPlugin, _ := newSpreadMemoPlugin(t, ctx, snapshot, feature.Features{}, spreadMemoNamespaces)
	incoming := spreadMemoIncoming("ns1", "incoming", nil)
	all := spreadMemoAllNodes

	runPreScoreSpread(t, ctx, memoPlugin, incoming, snapshot, all...)
	entry := onlySpreadEntry(t, memoPlugin)
	reused, recomputed := entry.memo.Reused, entry.memo.Recomputed
	if recomputed == 0 {
		t.Fatalf("the first pass recomputed nothing, so there is nothing to reuse either")
	}

	// Four of the six nodes contribute: node5 has no zone label, so requireAllTopologies excludes it
	// before any pod is counted, and node6 hosts nothing the constraints select. A memo that stored a zero contribution for it, or for a node whose pods
	// match nothing, would look correct and defeat the rule that lets a pass skip a node it already knows
	// to be empty.
	if got := entry.memo.Len(); got != 4 {
		t.Errorf("the first pass stored %d node contributions, want 4: only the nodes that count something may be stored", got)
	}
	// spreadMemoIncoming's second constraint is the hostname scoped one. Asserting on the aggregate
	// rather than on the state is the point: initPreScoreState never seeds a hostname constraint either,
	// so a memo that counted one would still produce the right state - after walking every pod of every
	// node to compute a count fill is going to drop.
	const hostnameConstraint = 1
	if got := len(entry.counts[hostnameConstraint]); got != 0 {
		t.Errorf("the memo holds %d topology values for the hostname scoped constraint, want 0: Score counts a node's own pods itself, so counting them here is work nobody reads", got)
	}

	runPreScoreSpread(t, ctx, memoPlugin, incoming, snapshot, all...)
	if entry.memo.Recomputed != recomputed {
		t.Errorf("an unmoved snapshot recomputed %d nodes, want 0", entry.memo.Recomputed-recomputed)
	}
	if entry.memo.Reused <= reused {
		t.Errorf("an unmoved snapshot reused %d nodes, want every stored node", entry.memo.Reused-reused)
	}

	// One node moves: exactly one node is recomputed.
	nodeInfo, err := snapshot.NodeInfos().Get("node3")
	if err != nil {
		t.Fatalf("Get(node3): %v", err)
	}
	moved := nodeInfo.Node().DeepCopy()
	moved.Labels["zone"] = "z3"
	nodeInfo.SetNode(moved)
	runPreScoreSpread(t, ctx, memoPlugin, incoming, snapshot, all...)
	if got := entry.memo.Recomputed - recomputed; got != 1 {
		t.Errorf("one moved node recomputed %d nodes, want 1", got)
	}
}

// TestSpreadKeyCoversItsInputs pins the key directly. A key that is coarser than the inputs is not a
// cache miss, it is one pod shape reading another shape's counts, and the walkthrough above cannot show
// it for inputs the fixture does not vary.
func TestSpreadKeyCoversItsInputs(t *testing.T) {
	selector := labels.SelectorFromSet(labels.Set{"app": "web"})
	constraint := func(topologyKey string, sel labels.Selector, maxSkew int32) topologySpreadConstraint {
		return topologySpreadConstraint{
			MaxSkew: maxSkew, TopologyKey: topologyKey, Selector: sel, MinDomains: 1,
			NodeAffinityPolicy: v1.NodeInclusionPolicyHonor, NodeTaintsPolicy: v1.NodeInclusionPolicyIgnore,
		}
	}
	constraints := []topologySpreadConstraint{constraint("zone", selector, 1)}

	pod := st.MakePod().Namespace("ns1").Name("incoming").UID("incoming").Label("app", "web").Obj()
	base := spreadKey(pod, constraints, true)

	otherNamespace := st.MakePod().Namespace("ns2").Name("incoming").UID("incoming").Label("app", "web").Obj()
	if got := spreadKey(otherNamespace, constraints, true); got == base {
		t.Errorf("another namespace does not change the spread key %q, and countPodsMatchSelector counts only the incoming pod's own namespace", got)
	}
	if got := spreadKey(pod, constraints, false); got == base {
		t.Errorf("requireAllTopologies does not change the spread key %q, and it decides whether a node missing a topology key contributes at all", got)
	}
	if got := spreadKey(pod, []topologySpreadConstraint{constraint("zone", selector, 3)}, true); got == base {
		t.Errorf("another MaxSkew does not change the spread key %q, and two constraints that differ in it are two constraints to the caller", got)
	}
	if got := spreadKey(pod, []topologySpreadConstraint{constraint("rack", selector, 1)}, true); got == base {
		t.Errorf("another topology key does not change the spread key %q", got)
	}
	if got := spreadKey(pod, []topologySpreadConstraint{constraint("zone", selector, 1), constraint("rack", selector, 1)}, true); got == base {
		t.Errorf("a second constraint does not change the spread key %q, and the aggregate is indexed by constraint", got)
	}
	// The same set of constraints in another order is another aggregate: the counts are indexed by
	// position, so serving one order's entry to the other would swap two constraints' counts.
	if got := spreadKey(pod, []topologySpreadConstraint{constraint("rack", selector, 1), constraint("zone", selector, 1)}, true); got == base {
		t.Errorf("reordering the constraints does not change the spread key %q, and the aggregate is indexed by position", got)
	}

	// A constraint with no labelSelector at all counts no pod; one with an empty labelSelector counts
	// every pod in the namespace. metav1.LabelSelectorAsSelector turns them into labels.Nothing and
	// labels.Everything, and Selector.String() renders both as "".
	nothing, err := metav1.LabelSelectorAsSelector(nil)
	if err != nil {
		t.Fatalf("LabelSelectorAsSelector(nil): %v", err)
	}
	everything, err := metav1.LabelSelectorAsSelector(&metav1.LabelSelector{})
	if err != nil {
		t.Fatalf("LabelSelectorAsSelector(empty): %v", err)
	}
	if !labels.MatchesNothing(nothing) || labels.MatchesNothing(everything) {
		t.Fatalf("the two selectors are supposed to be Nothing and Everything, got %v and %v", nothing, everything)
	}
	nothingKey := spreadKey(pod, []topologySpreadConstraint{constraint("zone", nothing, 1)}, true)
	everythingKey := spreadKey(pod, []topologySpreadConstraint{constraint("zone", everything, 1)}, true)
	if nothingKey == everythingKey {
		t.Errorf("a constraint that counts no pod and one that counts every pod share the spread key %q", nothingKey)
	}
	if nothingKey == base || everythingKey == base {
		t.Errorf("a constraint whose selector renders as the empty string shares the spread key %q with one that has a real selector", base)
	}

	// The node selector, the required node affinity and the tolerations are matched per node.
	withNodeSelector := pod.DeepCopy()
	withNodeSelector.Spec.NodeSelector = map[string]string{"zone": "z1"}
	if got := spreadKey(withNodeSelector, constraints, true); got == base {
		t.Errorf("a node selector does not change the spread key %q, and it excludes nodes from the walk", got)
	}
	// [term{A,B}] admits the nodes matching both A and B, [term{A},term{B}] those matching either. One
	// counts pods on nodes the other could never be placed on, so they must not share an entry.
	nodeAffinity := func(terms ...[]v1.NodeSelectorRequirement) *v1.Pod {
		p := pod.DeepCopy()
		p.Spec.Affinity = &v1.Affinity{NodeAffinity: &v1.NodeAffinity{
			RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
				NodeSelectorTerms: make([]v1.NodeSelectorTerm, 0, len(terms)),
			},
		}}
		for _, term := range terms {
			p.Spec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms =
				append(p.Spec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms,
					v1.NodeSelectorTerm{MatchExpressions: term})
		}
		return p
	}
	requirement := func(key, value string) v1.NodeSelectorRequirement {
		return v1.NodeSelectorRequirement{Key: key, Operator: v1.NodeSelectorOpIn, Values: []string{value}}
	}
	anded := nodeAffinity([]v1.NodeSelectorRequirement{requirement("zone", "z1"), requirement("rack", "r1")})
	ored := nodeAffinity([]v1.NodeSelectorRequirement{requirement("zone", "z1")}, []v1.NodeSelectorRequirement{requirement("rack", "r1")})
	if got, other := spreadKey(anded, constraints, true), spreadKey(ored, constraints, true); got == other {
		t.Errorf("a required node affinity whose terms are ANDed and one whose terms are ORed share the spread key %q, and the two admit opposite sets of nodes", got)
	}
	if got := spreadKey(anded, constraints, true); got == base {
		t.Errorf("a required node affinity does not change the spread key %q", got)
	}
	// A requirement in MatchFields is matched against the node's fields, and the same requirement in
	// MatchExpressions against its labels. The two admit different nodes, so they must not share a key -
	// and this is the pair only the tag separates, the term boundary being the same for both.
	withRequirement := func(field bool) *v1.Pod {
		p := pod.DeepCopy()
		term := v1.NodeSelectorTerm{}
		requirements := []v1.NodeSelectorRequirement{requirement("zone", "z1")}
		if field {
			term.MatchFields = requirements
		} else {
			term.MatchExpressions = requirements
		}
		p.Spec.Affinity = &v1.Affinity{NodeAffinity: &v1.NodeAffinity{
			RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{NodeSelectorTerms: []v1.NodeSelectorTerm{term}},
		}}
		return p
	}
	if got, other := spreadKey(withRequirement(true), constraints, true), spreadKey(withRequirement(false), constraints, true); got == other {
		t.Errorf("a requirement matched against the node's fields and the same requirement matched against its labels share the spread key %q", got)
	}
	if got := spreadKey(withRequirement(true), constraints, true); got == base {
		t.Errorf("a required node affinity on the node's fields does not change the spread key %q", got)
	}

	tolerating := pod.DeepCopy()
	tolerating.Spec.Tolerations = []v1.Toleration{{Key: "dedicated", Operator: v1.TolerationOpEqual, Value: "other", Effect: v1.TaintEffectNoSchedule}}
	if got := spreadKey(tolerating, constraints, true); got == base {
		t.Errorf("a toleration does not change the spread key %q, and a constraint with NodeTaintsPolicy=Honor matches a node's taints against it", got)
	}

	// The pod's own labels are deliberately NOT an input: nothing in the per node computation reads
	// them, the constraints were derived from them and are in the key themselves. Two workloads whose
	// pods spread the same way therefore share one entry, which is what makes the memo worth its
	// bookkeeping on a cluster that schedules more than one.
	otherLabels := pod.DeepCopy()
	otherLabels.Labels["pod-template-hash"] = "abc123"
	if got := spreadKey(otherLabels, constraints, true); got != base {
		t.Errorf("the pod's own labels changed the spread key from %q to %q, and nothing a node contributes depends on them", base, got)
	}
}

// TestPreScoreSkipsTheMemoInAPodGroupCycle is the reason PreScore looks at
// CycleState.IsPodGroupSchedulingCycle, and the PodTopologySpread counterpart of the InterPodAffinity
// test of the same name. A pod group cycle assumes every pod of the gang into one snapshot in place, and
// Snapshot.AssumePod puts the old generation back on purpose so that the snapshot stays consistent with
// the cache - which is exactly the one thing the memo cannot detect. The later pods of a gang therefore
// have to count from the snapshot itself, which is also what lets them see the earlier ones: a gang that
// scored on a memo would find every domain equally empty and land in one zone.
func TestPreScoreSkipsTheMemoInAPodGroupCycle(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	snapshot := spreadMemoFixture()
	memoPlugin, fullPlugin := newSpreadMemoPlugin(t, ctx, snapshot, feature.Features{}, spreadMemoNamespaces)
	all := spreadMemoAllNodes

	gangPod := func(name string) *v1.Pod {
		return spreadMemoIncoming("ns1", name, nil)
	}
	nodeInfoOf := func(name string) fwk.NodeInfo {
		nodeInfo, err := snapshot.NodeInfos().Get(name)
		if err != nil {
			t.Fatalf("Get(%s): %v", name, err)
		}
		return nodeInfo
	}

	// The first pod of the gang is scored in a normal cycle, so the memo is used and learns what node3
	// contributes to z2.
	first := gangPod("gang-0")
	runPreScoreSpread(t, ctx, memoPlugin, first, snapshot, all...)
	entry := onlySpreadEntry(t, memoPlugin)
	passesBefore := entry.memo.Reused + entry.memo.Recomputed
	z2Before := entry.counts[0]["z2"]
	if z2Before == 0 {
		t.Fatalf("the fixture puts nothing in z2, so the assumption below could not show up in a count")
	}

	// The scheduler now assumes a pod the constraints select onto node3, in place, the way a pod group
	// cycle does.
	assumed := st.MakePod().Namespace("ns1").Name("assumed").UID("assumed").Label("app", "web").Obj()
	assumed.Spec.NodeName = "node3"
	assumedInfo, err := framework.NewPodInfo(assumed)
	if err != nil {
		t.Fatalf("NewPodInfo: %v", err)
	}
	generationBefore := nodeInfoOf("node3").GetGeneration()
	if err := snapshot.AssumePod(assumedInfo); err != nil {
		t.Fatalf("AssumePod: %v", err)
	}
	if got := nodeInfoOf("node3").GetGeneration(); got != generationBefore {
		t.Fatalf("Snapshot.AssumePod moved node3's generation from %d to %d, so the test no longer models a pod group cycle",
			generationBefore, got)
	}

	// The second pod of the gang has to see the first one's assumption.
	second := gangPod("gang-1")
	state := framework.NewCycleState()
	state.SetPodGroupCycleState(state)
	if !state.IsPodGroupSchedulingCycle() {
		t.Fatalf("SetPodGroupCycleState did not make this a pod group cycle, so the test proves nothing")
	}
	filtered := make([]fwk.NodeInfo, 0, len(all))
	for _, name := range all {
		filtered = append(filtered, nodeInfoOf(name))
	}
	if status := memoPlugin.PreScore(ctx, state, second, filtered); !status.IsSuccess() {
		t.Fatalf("PreScore(gang-1): %v", status)
	}
	if passes := entry.memo.Reused + entry.memo.Recomputed; passes != passesBefore {
		t.Errorf("the pod group cycle consulted the scoring memo (%d node computations, want 0)", passes-passesBefore)
	}

	got, err := getPreScoreState(state)
	if err != nil {
		t.Fatalf("getPreScoreState: %v", err)
	}
	want := runPreScoreSpread(t, ctx, fullPlugin, second, snapshot, all...)
	gotCounts := make([]map[string]int64, len(got.TopologyValueToPodCounts))
	for i, byValue := range got.TopologyValueToPodCounts {
		gotCounts[i] = make(map[string]int64, len(byValue))
		for value, count := range byValue {
			gotCounts[i][value] = *count
		}
	}
	if diff := cmp.Diff(want.Counts, gotCounts); diff != "" {
		t.Errorf("the gang's second pod counted differently from the full computation (-want,+got):\n%s", diff)
	}
	if gotCounts[0]["z2"] <= z2Before {
		t.Errorf("the gang's second pod counted %d pods in z2, which is not more than the %d there before the assumption: the pod assumed onto node3 during the gang cycle was not counted, so a gang would pile into one zone",
			gotCounts[0]["z2"], z2Before)
	}
}
