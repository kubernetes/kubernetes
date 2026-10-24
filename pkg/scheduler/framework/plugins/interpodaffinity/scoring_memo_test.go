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
	"testing"

	"github.com/google/go-cmp/cmp"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/backend/cache"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodememo"
	plugintesting "k8s.io/kubernetes/pkg/scheduler/framework/plugins/testing"
	schedruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

// scoringResult is what a PreScore run leaves behind, reduced to what the memo produces and to what
// the framework goes on to do with it. The rest of preScoreState - the pod info, the namespace labels
// - is copied through untouched, and comparing it would need cmp options for framework.PodInfo's
// unexported fields without saying anything about the memo.
type scoringResult struct {
	Skipped bool
	// TopologyScore is the aggregate with its zero valued entries dropped, which is the one way the
	// two paths are allowed to differ: add drops a topology value that reaches zero so that the
	// aggregate cannot grow without bound, and a zero cannot change a score. See scoreMap.add.
	TopologyScore map[string]map[string]int64
	// Scores and Normalized are what Score and NormalizeScore make of it, node by node. Comparing them
	// is the assertion that actually matters; comparing the map is the one that localizes a failure.
	Scores     map[string]int64
	Normalized map[string]int64
}

// normalizeScoreMap drops the zero valued entries of a scoreMap so that cmp can diff the two paths'
// aggregates without being told about anything unexported.
func normalizeScoreMap(m scoreMap) map[string]map[string]int64 {
	out := make(map[string]map[string]int64, len(m))
	for topology, values := range m {
		for value, score := range values {
			if score == 0 {
				continue
			}
			if out[topology] == nil {
				out[topology] = make(map[string]int64)
			}
			out[topology][value] = score
		}
	}
	return out
}

// newScoringMemoPlugin returns two plugins over one snapshot: the one New builds, which memoizes, and
// a copy of it with the scoring memo taken out, which therefore takes the full computation. Everything
// else about the two is identical, so a difference between them is a difference the memo made.
func newScoringMemoPlugin(t *testing.T, ctx context.Context, snapshot *cache.Snapshot,
	namespaces []runtime.Object) (memo, full *InterPodAffinity) {
	t.Helper()
	p := plugintesting.SetupPluginWithInformers(ctx, t,
		schedruntime.FactoryAdapter(feature.Features{}, New),
		&config.InterPodAffinityArgs{}, snapshot, namespaces)
	memoPlugin := p.(*InterPodAffinity)
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

// runPreScore drives one plugin through PreScore, Score and NormalizeScore over every node of the
// snapshot, and returns everything the framework would go on to use.
func runPreScore(t *testing.T, ctx context.Context, pl *InterPodAffinity, pod *v1.Pod, snapshot *cache.Snapshot) scoringResult {
	t.Helper()
	nodes, err := snapshot.NodeInfos().List()
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	state := framework.NewCycleState()
	status := pl.PreScore(ctx, state, pod, nodes)
	if status != nil && status.Code() == fwk.Skip {
		return scoringResult{Skipped: true}
	}
	if !status.IsSuccess() {
		t.Fatalf("PreScore(%s): %v", pod.Name, status)
	}
	s, err := getPreScoreState(state)
	if err != nil {
		t.Fatalf("getPreScoreState(%s): %v", pod.Name, err)
	}

	result := scoringResult{
		TopologyScore: normalizeScoreMap(s.topologyScore),
		Scores:        make(map[string]int64, len(nodes)),
		Normalized:    make(map[string]int64, len(nodes)),
	}
	scoreList := make(fwk.NodeScoreList, 0, len(nodes))
	for _, nodeInfo := range nodes {
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

// scoringMemoFixture builds three nodes in two zones and pods that make every branch of
// processExistingPod contribute, so that a memo that got one of them wrong could not hide behind the
// others:
//   - node1 hosts a pod with a preferred anti-affinity term matching the incoming pod, which is the
//     existing pod's own soft terms, and carries app=target so the incoming pod's soft terms match it
//     too - one negative contribution from each half;
//   - node2 hosts two pods with a required affinity term matching the incoming pod, which is the
//     args.HardPodAffinityWeight branch, and both carry app=target, so the incoming pod's soft
//     anti-affinity subtracts exactly what the hard branch adds. That node's contribution is a
//     topology value of zero, which is the case scoreMap.add has to keep out of the aggregate without
//     losing the fact that the node contributed;
//   - node3 hosts a plain pod, which only the incoming pod's own terms can match.
func scoringMemoFixture() (*cache.Snapshot, []*v1.Node) {
	nodes := []*v1.Node{
		st.MakeNode().Name("node1").Label("zone", "z1").Label(v1.LabelHostname, "node1").Obj(),
		st.MakeNode().Name("node2").Label("zone", "z1").Label(v1.LabelHostname, "node2").Obj(),
		st.MakeNode().Name("node3").Label("zone", "z2").Label(v1.LabelHostname, "node3").Obj(),
	}
	pods := []*v1.Pod{
		st.MakePod().Namespace("ns1").Name("soft-on-node1").UID("soft-on-node1").
			Label("app", "target").Node("node1").
			PodAntiAffinityExists("app", "zone", st.PodAntiAffinityWithPreferredReq).Obj(),
		st.MakePod().Namespace("ns1").Name("hard-a-on-node2").UID("hard-a-on-node2").
			Label("app", "target").Node("node2").
			PodAffinityExists("victim", "zone", st.PodAffinityWithRequiredReq).Obj(),
		st.MakePod().Namespace("ns1").Name("hard-b-on-node2").UID("hard-b-on-node2").
			Label("app", "target").Node("node2").
			PodAffinityExists("victim", "zone", st.PodAffinityWithRequiredReq).Obj(),
		st.MakePod().Namespace("ns1").Name("plain-on-node3").UID("plain-on-node3").
			Label("app", "target").Node("node3").Obj(),
	}
	return cache.NewSnapshot(pods, nodes), nodes
}

// scoringMemoPreferred is an incoming pod with a preferred zone scoped anti-affinity term selecting
// anything labeled app, so hasConstraints is true and PreScore walks every node and every pod.
func scoringMemoPreferred(namespace, name string, mutate func(*v1.Pod)) *v1.Pod {
	pod := st.MakePod().Namespace(namespace).Name(name).UID(name).Label("victim", "yes").Obj()
	pod.Labels["app"] = "victim"
	pod.Spec.Affinity = &v1.Affinity{
		PodAntiAffinity: &v1.PodAntiAffinity{
			PreferredDuringSchedulingIgnoredDuringExecution: []v1.WeightedPodAffinityTerm{{
				Weight: 1,
				PodAffinityTerm: v1.PodAffinityTerm{
					TopologyKey:   "zone",
					LabelSelector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": "target"}},
				},
			}},
		},
	}
	if mutate != nil {
		mutate(pod)
	}
	return pod
}

// scoringMemoRequired is an incoming pod with only required terms: hasConstraints is false, so PreScore
// walks the snapshot's HavePodsWithAffinityList and, per node, only its pods with affinity. Everything
// it can score comes from the existing pods' own terms matched back against it.
func scoringMemoRequired(namespace, name string) *v1.Pod {
	pod := st.MakePod().Namespace(namespace).Name(name).UID(name).Label("app", "victim").Obj()
	pod.Spec.Affinity = &v1.Affinity{
		PodAffinity: &v1.PodAffinity{
			RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{{
				TopologyKey:   "zone",
				LabelSelector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": "target"}},
			}},
		},
	}
	return pod
}

func TestScoringMemoMatchesFullComputation(t *testing.T) {
	scoringMemoMatchesFullComputation(t, nodememo.DefaultParallelMinNodes)
}

// TestScoringMemoMatchesFullComputationInParallel runs the same steps with the parallel threshold at
// zero, so every pass spreads the matching over the parallelizer and has to land on the same numbers.
func TestScoringMemoMatchesFullComputationInParallel(t *testing.T) {
	scoringMemoMatchesFullComputation(t, 0)
}

func scoringMemoMatchesFullComputation(t *testing.T, parallelMinNodes int) {
	previous := memoParallelMinNodes
	memoParallelMinNodes = parallelMinNodes
	defer func() { memoParallelMinNodes = previous }()

	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	snapshot, _ := scoringMemoFixture()
	namespaces := []runtime.Object{
		&v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: "ns1", Labels: map[string]string{"team": "infra"}}},
	}
	memoPlugin, fullPlugin := newScoringMemoPlugin(t, ctx, snapshot, namespaces)

	// Both shapes are compared at every step: they walk different node lists and process different pods
	// per node, so they are two different memo entries and two different sets of contributions.
	preferred := scoringMemoPreferred("ns1", "incoming-soft", nil)
	required := scoringMemoRequired("ns1", "incoming-hard")

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
	podNamed := func(node, name string) *v1.Pod {
		t.Helper()
		for _, podInfo := range nodeInfoOf(node).GetPods() {
			if podInfo.GetPod().Name == name {
				return podInfo.GetPod()
			}
		}
		t.Fatalf("no pod %s on %s", name, node)
		return nil
	}
	relabelNode := func(name, zone string) {
		t.Helper()
		node := nodeInfoOf(name).Node().DeepCopy()
		node.Labels["zone"] = zone
		nodeInfoOf(name).SetNode(node)
	}

	step := func(name string) {
		t.Helper()
		for _, pod := range []*v1.Pod{preferred, required} {
			want := runPreScore(t, ctx, fullPlugin, pod, snapshot)
			got := runPreScore(t, ctx, memoPlugin, pod, snapshot)
			if diff := cmp.Diff(want, got); diff != "" {
				t.Errorf("%s/%s: the memo's scores differ from the full computation (-want,+got):\n%s",
					name, pod.Name, diff)
			}
			if want.Skipped {
				t.Fatalf("%s/%s: PreScore Skipped, so the step proves nothing about the memo", name, pod.Name)
			}
			if len(want.TopologyScore) == 0 {
				t.Fatalf("%s/%s: the fixture produced no topology scores at all", name, pod.Name)
			}
		}
	}

	step("initial")

	// A pod the incoming pod's own soft anti-affinity matches appears on the node that had none.
	added := st.MakePod().Namespace("ns1").Name("added-on-node3").UID("added-on-node3").
		Label("app", "target").Node("node3").
		PodAntiAffinityExists("app", "zone", st.PodAntiAffinityWithPreferredReq).Obj()
	addPod(added)
	step("pod added")

	// ... and disappears again, which is the half that needs subtract: a memo that could only add would
	// keep counting it.
	removePod(added)
	step("pod removed")

	// node3 moves to another topology domain, which moves every score keyed by zone.
	relabelNode("node3", "z3")
	step("node relabelled")
	relabelNode("node3", "z2")
	step("node relabelled back")

	// A pod with a required affinity term joins the node that only had a soft one, so node1 starts
	// appearing in HavePodsWithAffinityList for a second reason and the hard branch has something to add
	// there too.
	hardOnNode1 := st.MakePod().Namespace("ns1").Name("hard-on-node1").UID("hard-on-node1").
		Label("app", "target").Node("node1").
		PodAffinityExists("victim", "zone", st.PodAffinityWithRequiredReq).Obj()
	addPod(hardOnNode1)
	step("pod with required affinity added")

	// One of node2's two hard pods goes away, so its contribution stops canceling out and the aggregate
	// has to grow a topology value it never had a non-zero one for.
	removePod(podNamed("node2", "hard-a-on-node2"))
	step("one of two canceling pods removed")
}

// onlyScoringEntry returns the single entry a test's pod shape is stored under.
func onlyScoringEntry(t *testing.T, pl *InterPodAffinity) *scoringEntry {
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

// TestScoringMemoReusesUnchangedNodes is the other half: correctness alone would also be served by
// recomputing every node every cycle, which is what the memo exists to avoid.
func TestScoringMemoReusesUnchangedNodes(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	snapshot, _ := scoringMemoFixture()
	namespaces := []runtime.Object{
		&v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: "ns1", Labels: map[string]string{"team": "infra"}}},
	}
	memoPlugin, _ := newScoringMemoPlugin(t, ctx, snapshot, namespaces)
	incoming := scoringMemoPreferred("ns1", "incoming", nil)

	runPreScore(t, ctx, memoPlugin, incoming, snapshot)
	entry := onlyScoringEntry(t, memoPlugin)
	reused, recomputed := entry.memo.Reused, entry.memo.Recomputed
	if recomputed == 0 {
		t.Fatalf("the first pass recomputed nothing, so there is nothing to reuse either")
	}

	// The same shape over an unmoved snapshot: every stored node is reused, nothing recomputed.
	runPreScore(t, ctx, memoPlugin, incoming, snapshot)
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
	runPreScore(t, ctx, memoPlugin, incoming, snapshot)
	if got := entry.memo.Recomputed - recomputed; got != 1 {
		t.Errorf("one moved node recomputed %d nodes, want 1", got)
	}
}

// TestScoringMemoKeepsTheSkipDecisionWhenContributionsCancel is what scoringEntry.contributing is for.
// PreScore Skips when no node contributed anything, which is not the same as the aggregate being empty:
// a node whose pods add and subtract the same weight contributes a topology value of zero, and add
// drops zeros so that the aggregate cannot grow without bound. Deciding Skip off len(topologyScore)
// would then Skip a pod the full computation scores, and hand the framework no scores at all for a
// plugin it was told to run.
func TestScoringMemoKeepsTheSkipDecisionWhenContributionsCancel(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	// One node, one pod that the incoming pod's soft affinity and soft anti-affinity both match with the
	// same weight, so the only contribution in the cluster is +1 and -1 to the same topology value.
	nodes := []*v1.Node{st.MakeNode().Name("node1").Label("zone", "z1").Label(v1.LabelHostname, "node1").Obj()}
	pods := []*v1.Pod{
		st.MakePod().Namespace("ns1").Name("target").UID("target").Label("app", "target").Node("node1").Obj(),
	}
	snapshot := cache.NewSnapshot(pods, nodes)
	namespaces := []runtime.Object{
		&v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: "ns1", Labels: map[string]string{"team": "infra"}}},
	}
	memoPlugin, fullPlugin := newScoringMemoPlugin(t, ctx, snapshot, namespaces)

	canceling := scoringMemoPreferred("ns1", "canceling", func(pod *v1.Pod) {
		term := pod.Spec.Affinity.PodAntiAffinity.PreferredDuringSchedulingIgnoredDuringExecution[0].PodAffinityTerm
		pod.Spec.Affinity.PodAffinity = &v1.PodAffinity{
			PreferredDuringSchedulingIgnoredDuringExecution: []v1.WeightedPodAffinityTerm{{Weight: 1, PodAffinityTerm: term}},
		}
	})

	want := runPreScore(t, ctx, fullPlugin, canceling, snapshot)
	if want.Skipped {
		t.Fatalf("the full computation Skipped, so the fixture does not produce a canceling contribution and the test proves nothing")
	}
	got := runPreScore(t, ctx, memoPlugin, canceling, snapshot)
	if got.Skipped {
		t.Fatalf("the memo Skipped where the full computation scored: contributions that cancel each other still contributed")
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("the memo's scores differ from the full computation (-want,+got):\n%s", diff)
	}
	entry := onlyScoringEntry(t, memoPlugin)
	if entry.contributing != 1 {
		t.Errorf("the memo counts %d contributing nodes, want 1", entry.contributing)
	}

	// The counter's other half: when the canceling pod goes away the node stops contributing at all,
	// and both paths have to Skip. A memo that could increment the counter but not decrement it would
	// keep scoring a cluster with nothing left to score against.
	target := pods[0]
	nodeInfo, err := snapshot.NodeInfos().Get("node1")
	if err != nil {
		t.Fatalf("Get(node1): %v", err)
	}
	if err := nodeInfo.RemovePod(klog.FromContext(ctx), target); err != nil {
		t.Fatalf("RemovePod: %v", err)
	}
	want = runPreScore(t, ctx, fullPlugin, canceling, snapshot)
	got = runPreScore(t, ctx, memoPlugin, canceling, snapshot)
	if !want.Skipped {
		t.Fatalf("the full computation still scores an empty cluster, so the fixture no longer proves anything")
	}
	if !got.Skipped {
		t.Errorf("the memo still scores an empty cluster: the node that stopped contributing was not withdrawn")
	}
	if entry.contributing != 0 {
		t.Errorf("the memo counts %d contributing nodes after the only pod was removed, want 0", entry.contributing)
	}
}

// TestScoringMemoKeysCoverTheirInputs pins the key directly. A key that is coarser than the inputs is
// not a cache miss, it is one pod shape reading another shape's scores, and the walkthrough above
// cannot show it for inputs the fixture does not vary - the labels of a namespace, which only a
// namespaceSelector term is matched against, the weight of a preferred term, and the two selectors
// that render alike while meaning opposite things.
func TestScoringMemoKeysCoverTheirInputs(t *testing.T) {
	nsLabels := labels.Set{"team": "infra"}
	otherNsLabels := labels.Set{"team": "prod"}

	stateOf := func(t *testing.T, pod *v1.Pod, nsLabels labels.Set) *preScoreState {
		t.Helper()
		podInfo, err := framework.NewPodInfo(pod)
		if err != nil {
			t.Fatalf("NewPodInfo: %v", err)
		}
		return &preScoreState{podInfo: podInfo, namespaceLabels: nsLabels}
	}

	inNS1 := scoringMemoPreferred("ns1", "incoming", nil)
	// Same pod labels, a namespace whose labels are identical to ns1's: nothing but the namespace itself
	// can separate these two keys.
	inNS3 := scoringMemoPreferred("ns3", "incoming", nil)
	extraLabel := scoringMemoPreferred("ns1", "incoming", func(pod *v1.Pod) { pod.Labels["extra"] = "yes" })
	heavier := scoringMemoPreferred("ns1", "incoming", func(pod *v1.Pod) {
		pod.Spec.Affinity.PodAntiAffinity.PreferredDuringSchedulingIgnoredDuringExecution[0].Weight = 7
	})
	asAffinity := scoringMemoPreferred("ns1", "incoming", func(pod *v1.Pod) {
		term := pod.Spec.Affinity.PodAntiAffinity.PreferredDuringSchedulingIgnoredDuringExecution[0]
		pod.Spec.Affinity.PodAntiAffinity = nil
		pod.Spec.Affinity.PodAffinity = &v1.PodAffinity{
			PreferredDuringSchedulingIgnoredDuringExecution: []v1.WeightedPodAffinityTerm{term},
		}
	})

	state := stateOf(t, inNS1, nsLabels)
	for _, tc := range []struct {
		name       string
		other      string
		otherState *preScoreState
		otherPod   *v1.Pod
	}{
		{"another namespace with identical labels", inNS3.Name, stateOf(t, inNS3, nsLabels), inNS3},
		{"another set of pod labels", extraLabel.Name, stateOf(t, extraLabel, nsLabels), extraLabel},
		{"a heavier preferred term", heavier.Name, stateOf(t, heavier, nsLabels), heavier},
		{"the same term as an affinity instead", asAffinity.Name, stateOf(t, asAffinity, nsLabels), asAffinity},
	} {
		if got, other := scoringKey(inNS1, state, true), scoringKey(tc.otherPod, tc.otherState, true); got == other {
			t.Errorf("%s does not change the scoring key %q", tc.name, got)
		}
	}
	if got, other := scoringKey(inNS1, state, true), scoringKey(inNS1, state, false); got == other {
		t.Errorf("hasConstraints does not change the scoring key %q, and it picks both the node list and which pods of a node are scored", got)
	}
	if got, other := scoringKey(inNS1, state, true), scoringKey(inNS1, stateOf(t, inNS1, otherNsLabels), true); got == other {
		t.Errorf("relabeling a namespace does not change the scoring key %q, and no node generation moves when a namespace is relabeled", got)
	}
	if got, other := scoringKey(inNS1, state, true), scoringKey(inNS1, stateOf(t, scoringMemoRequired("ns1", "incoming"), nsLabels), true); got == other {
		t.Errorf("dropping the preferred terms does not change the scoring key %q", got)
	}

	// A term with no labelSelector at all matches no pod; a term with an empty one matches every pod in
	// its namespaces. metav1.LabelSelectorAsSelector turns them into labels.Nothing and labels.Everything,
	// and Selector.String() renders both as "".
	noSelector := func(pod *v1.Pod) {
		pod.Spec.Affinity.PodAntiAffinity.PreferredDuringSchedulingIgnoredDuringExecution[0].PodAffinityTerm.LabelSelector = nil
	}
	emptySelector := func(pod *v1.Pod) {
		pod.Spec.Affinity.PodAntiAffinity.PreferredDuringSchedulingIgnoredDuringExecution[0].PodAffinityTerm.LabelSelector = &metav1.LabelSelector{}
	}
	nothingState := stateOf(t, scoringMemoPreferred("ns1", "incoming", noSelector), nsLabels)
	everythingState := stateOf(t, scoringMemoPreferred("ns1", "incoming", emptySelector), nsLabels)
	if terms := nothingState.podInfo.GetPreferredAntiAffinityTerms(); len(terms) != 1 || !labels.MatchesNothing(terms[0].Selector) {
		t.Fatalf("a term with no labelSelector is supposed to be labels.Nothing(), got %+v", terms)
	}
	if terms := everythingState.podInfo.GetPreferredAntiAffinityTerms(); len(terms) != 1 || labels.MatchesNothing(terms[0].Selector) {
		t.Fatalf("a term with an empty labelSelector is supposed to be labels.Everything(), got %+v", terms)
	}
	if got, other := scoringKey(inNS1, nothingState, true), scoringKey(inNS1, everythingState, true); got == other {
		t.Errorf("a term that matches no pod and one that matches every pod share the scoring key %q", got)
	}
}

// TestPreScoreSkipsTheMemoInAPodGroupCycle is the reason PreScore looks at
// CycleState.IsPodGroupSchedulingCycle, and the scoring counterpart of the filtering test of the same
// name. A pod group cycle assumes every pod of the gang into one snapshot in place, and
// Snapshot.AssumePod puts the old generation back on purpose so that the snapshot stays consistent with
// the cache - which is exactly the one thing the memo cannot detect. The later pods of a gang therefore
// have to score from the snapshot itself, which is also what lets them see the earlier ones.
func TestPreScoreSkipsTheMemoInAPodGroupCycle(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	snapshot, _ := scoringMemoFixture()
	namespaces := []runtime.Object{
		&v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: "ns1", Labels: map[string]string{"team": "infra"}}},
	}
	memoPlugin, fullPlugin := newScoringMemoPlugin(t, ctx, snapshot, namespaces)

	gangPod := func(name string) *v1.Pod {
		return scoringMemoPreferred("ns1", name, nil)
	}
	nodeInfoOf := func(name string) fwk.NodeInfo {
		nodeInfo, err := snapshot.NodeInfos().Get(name)
		if err != nil {
			t.Fatalf("Get(%s): %v", name, err)
		}
		return nodeInfo
	}

	// The first pod of the gang is scored in a normal cycle, so the memo is used and learns what node3
	// contributes.
	first := gangPod("gang-0")
	runPreScore(t, ctx, memoPlugin, first, snapshot)
	entry := onlyScoringEntry(t, memoPlugin)
	passesBefore := entry.memo.Reused + entry.memo.Recomputed
	before := entry.topologyScore["zone"]["z2"]

	// The scheduler now assumes a pod the gang's soft anti-affinity selects onto node3, in place, the way
	// a pod group cycle does.
	assumed := st.MakePod().Namespace("ns1").Name("assumed").UID("assumed").Label("app", "target").Obj()
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
	nodes, err := snapshot.NodeInfos().List()
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if status := memoPlugin.PreScore(ctx, state, second, nodes); !status.IsSuccess() {
		t.Fatalf("PreScore(gang-1): %v", status)
	}
	if passes := entry.memo.Reused + entry.memo.Recomputed; passes != passesBefore {
		t.Errorf("the pod group cycle consulted the scoring memo (%d node computations, want 0)", passes-passesBefore)
	}

	got, err := getPreScoreState(state)
	if err != nil {
		t.Fatalf("getPreScoreState: %v", err)
	}
	want := runPreScore(t, ctx, fullPlugin, second, snapshot)
	if diff := cmp.Diff(want.TopologyScore, normalizeScoreMap(got.topologyScore)); diff != "" {
		t.Errorf("the gang's second pod scored differently from the full computation (-want,+got):\n%s", diff)
	}
	if got.topologyScore["zone"]["z2"] >= before {
		t.Errorf("the gang's second pod scored zone z2 as %d, which is not below the %d it was before the assumption: the pod assumed onto node3 during the gang cycle was not counted, so a gang would pile into one zone",
			got.topologyScore["zone"]["z2"], before)
	}
}
