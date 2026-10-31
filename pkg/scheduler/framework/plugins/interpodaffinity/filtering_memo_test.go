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
	"testing"

	"github.com/google/go-cmp/cmp"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/backend/cache"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodememo"
	plugintesting "k8s.io/kubernetes/pkg/scheduler/framework/plugins/testing"
	schedruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

// filteringCounts is what a PreFilter run leaves behind, reduced to the numbers the memo produces. The
// rest of preFilterState - the pod info, the classified terms - is copied through untouched, and
// comparing it would need cmp options for framework.PodInfo's unexported fields without saying
// anything about the memo.
type filteringCounts struct {
	Skipped                    bool
	ExistingClusterWide        map[string]int64
	ClusterWideAffinity        map[string]int64
	ClusterWideAntiAffinity    map[string]int64
	MatchingHostScopedAffinity int64
}

// renderCounts flattens a count map into something cmp can diff without being told about
// topologyPair's unexported fields, and whose diff is readable.
func renderCounts(counts topologyToMatchedTermCount) map[string]int64 {
	out := make(map[string]int64, len(counts))
	for pair, count := range counts {
		out[pair.key+"="+pair.value] = count
	}
	return out
}

// newFilteringMemoPlugin returns two plugins over one snapshot: the one New builds, which memoizes,
// and a copy of it with the memos taken out, which therefore takes the full computation. Everything
// else about the two is identical, so a difference between them is a difference the memo made.
func newFilteringMemoPlugin(t *testing.T, ctx context.Context, snapshot *cache.Snapshot, fastPath bool,
	namespaces []runtime.Object) (memo, full *InterPodAffinity) {
	t.Helper()
	p := plugintesting.SetupPluginWithInformers(ctx, t,
		schedruntime.FactoryAdapter(feature.Features{EnableInterPodAffinityHostnameFastPath: fastPath}, New),
		&config.InterPodAffinityArgs{}, snapshot, namespaces)
	memoPlugin := p.(*InterPodAffinity)
	if !memoPlugin.filteringMemoReady() {
		t.Fatalf("New did not build the filtering memos")
	}
	fullPlugin := *memoPlugin
	fullPlugin.filteringExistingMemo = nil
	fullPlugin.filteringIncomingMemo = nil
	fullPlugin.filteringHostScopedAffinityMemo = nil
	if fullPlugin.filteringMemoReady() {
		t.Fatalf("the plugin meant to take the full computation still has memos")
	}
	return memoPlugin, &fullPlugin
}

// runPreFilter returns the counts one plugin's PreFilter produced for pod over this snapshot.
func runPreFilter(t *testing.T, ctx context.Context, pl *InterPodAffinity, pod *v1.Pod, snapshot *cache.Snapshot) filteringCounts {
	t.Helper()
	nodes, err := snapshot.NodeInfos().List()
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	state := framework.NewCycleState()
	_, status := pl.PreFilter(ctx, state, pod, nodes)
	if status != nil && status.Code() == fwk.Skip {
		return filteringCounts{Skipped: true}
	}
	if !status.IsSuccess() {
		t.Fatalf("PreFilter(%s): %v", pod.Name, status)
	}
	s, err := getPreFilterState(state)
	if err != nil {
		t.Fatalf("getPreFilterState(%s): %v", pod.Name, err)
	}
	return filteringCounts{
		ExistingClusterWide:        renderCounts(s.existingClusterWideAntiAffinityCounts),
		ClusterWideAffinity:        renderCounts(s.clusterWideAffinityCounts),
		ClusterWideAntiAffinity:    renderCounts(s.clusterWideAntiAffinityCounts),
		MatchingHostScopedAffinity: s.matchingHostScopedAffinityPodsCount,
	}
}

// filteringMemoFixture builds three nodes in two zones and the pods that make all three cluster wide
// walks of PreFilter produce something:
//   - node1 hosts a pod with a required zone scoped anti-affinity term, so node1 is in both of the
//     snapshot's filtered anti-affinity lists and the existing pods' counts are non-empty;
//   - node2 hosts a pod with a required hostname scoped anti-affinity term, which is the case the
//     hostname fast path splits off into the host scoped flag instead of the counts;
//   - node3 hosts plain pods, which the incoming pod's own terms match.
func filteringMemoFixture() (*cache.Snapshot, []*v1.Node) {
	nodes := []*v1.Node{
		st.MakeNode().Name("node1").Label("zone", "z1").Label(v1.LabelHostname, "node1").Obj(),
		st.MakeNode().Name("node2").Label("zone", "z1").Label(v1.LabelHostname, "node2").Obj(),
		st.MakeNode().Name("node3").Label("zone", "z2").Label(v1.LabelHostname, "node3").Obj(),
	}
	pods := []*v1.Pod{
		// Required anti-affinity, zone scoped, against anything labeled app=victim.
		st.MakePod().Namespace("ns1").Name("anti-zone-on-node1").UID("anti-zone-on-node1").
			Label("owner", "yes").Node("node1").
			PodAntiAffinityExists("app", "zone", st.PodAntiAffinityWithRequiredReq).Obj(),
		// Required anti-affinity, hostname scoped: the fast path counts it as a flag, not as a count.
		st.MakePod().Namespace("ns1").Name("anti-host-on-node2").UID("anti-host-on-node2").
			Label("owner", "yes").Node("node2").
			PodAntiAffinityExists("app", v1.LabelHostname, st.PodAntiAffinityWithRequiredReq).Obj(),
		// And a zone scoped one on the same node whose selector the incoming pod does not match. This is
		// what puts node2 in HavePodsWithRequiredNonHostScopedAntiAffinityList, so that under the fast
		// path node2 is walked and contributes the host scoped flag and nothing else - the case a memo
		// that stored only nodes with counts would silently drop.
		st.MakePod().Namespace("ns1").Name("anti-zone-nomatch-on-node2").UID("anti-zone-nomatch-on-node2").
			Label("owner", "yes").Node("node2").
			PodAntiAffinityExists("nomatch", "zone", st.PodAntiAffinityWithRequiredReq).Obj(),
		st.MakePod().Namespace("ns1").Name("victim-on-node3").UID("victim-on-node3").
			Label("app", "victim").Node("node3").Obj(),
		st.MakePod().Namespace("ns1").Name("target-on-node3").UID("target-on-node3").
			Label("app", "target").Node("node3").Obj(),
	}
	return cache.NewSnapshot(pods, nodes), nodes
}

// filteringMemoIncoming is the incoming pod: a required zone scoped affinity term selecting app=target
// and a required zone scoped anti-affinity term selecting app=anti, so that both halves of
// getIncomingAffinityAntiAffinityCounts have something to find.
func filteringMemoIncoming(namespace, name string, mutate func(*v1.Pod)) *v1.Pod {
	selector := func(key, value string) *metav1.LabelSelector {
		return &metav1.LabelSelector{MatchLabels: map[string]string{key: value}}
	}
	pod := st.MakePod().Namespace(namespace).Name(name).UID(name).Label("app", "victim").Obj()
	pod.Spec.Affinity = &v1.Affinity{
		PodAffinity: &v1.PodAffinity{
			RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{{
				TopologyKey: "zone", LabelSelector: selector("app", "target"),
			}},
		},
		PodAntiAffinity: &v1.PodAntiAffinity{
			RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{{
				TopologyKey: "zone", LabelSelector: selector("app", "anti"),
			}},
		},
	}
	if mutate != nil {
		mutate(pod)
	}
	return pod
}

func TestFilteringMemoMatchesFullComputation(t *testing.T) {
	for _, fastPath := range []bool{false, true} {
		t.Run(fmt.Sprintf("InterPodAffinityHostnameFastPath=%v", fastPath), func(t *testing.T) {
			filteringMemoMatchesFullComputation(t, fastPath, nodememo.DefaultParallelMinNodes)
		})
	}
}

// TestFilteringMemoMatchesFullComputationInParallel runs the same steps with the parallel threshold at
// zero, so every pass spreads the matching over the parallelizer and has to land on the same numbers.
func TestFilteringMemoMatchesFullComputationInParallel(t *testing.T) {
	for _, fastPath := range []bool{false, true} {
		t.Run(fmt.Sprintf("InterPodAffinityHostnameFastPath=%v", fastPath), func(t *testing.T) {
			filteringMemoMatchesFullComputation(t, fastPath, 0)
		})
	}
}

func filteringMemoMatchesFullComputation(t *testing.T, fastPath bool, parallelMinNodes int) {
	previous := memoParallelMinNodes
	memoParallelMinNodes = parallelMinNodes
	defer func() { memoParallelMinNodes = previous }()

	// The gate has to be set globally, not just handed to the plugin: the snapshot decides which of its
	// filtered anti-affinity lists to populate while it is being built, so a snapshot built with the gate
	// off has an empty HavePodsWithRequiredNonHostScopedAntiAffinityList and the fast path would walk
	// nothing.
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate,
		features.InterPodAffinityHostnameFastPath, fastPath)

	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	snapshot, _ := filteringMemoFixture()
	namespaces := []runtime.Object{
		&v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: "ns1", Labels: map[string]string{"team": "infra"}}},
		&v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: "ns2", Labels: map[string]string{"team": "prod"}}},
		// ns3 carries the same labels as ns1 on purpose: a pod of the same shape in ns3 is then
		// indistinguishable from one in ns1 by its labels and by its namespace labels alike, and only the
		// namespace itself can tell the two memo keys apart.
		&v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: "ns3", Labels: map[string]string{"team": "infra"}}},
	}
	memoPlugin, fullPlugin := newFilteringMemoPlugin(t, ctx, snapshot, fastPath, namespaces)

	incoming := filteringMemoIncoming("ns1", "incoming", nil)
	nodeInfoOf := func(name string) fwk.NodeInfo {
		nodeInfo, err := snapshot.NodeInfos().Get(name)
		if err != nil {
			t.Fatalf("Get(%s): %v", name, err)
		}
		return nodeInfo
	}
	addPod := func(pod *v1.Pod) {
		podInfo, err := framework.NewPodInfo(pod)
		if err != nil {
			t.Fatalf("NewPodInfo(%s): %v", pod.Name, err)
		}
		nodeInfoOf(pod.Spec.NodeName).AddPodInfo(podInfo)
	}
	removePod := func(pod *v1.Pod) {
		if err := nodeInfoOf(pod.Spec.NodeName).RemovePod(klog.FromContext(ctx), pod); err != nil {
			t.Fatalf("RemovePod(%s): %v", pod.Name, err)
		}
	}
	relabelNode := func(name, zone string) {
		node := nodeInfoOf(name).Node().DeepCopy()
		node.Labels["zone"] = zone
		nodeInfoOf(name).SetNode(node)
	}

	// existingFlag compares the second return of the existing pods' counts directly, because PreFilter
	// folds it into a local and never puts it in the cycle state: with the hostname fast path on it is
	// what decides whether host scoped anti-affinity still has to be evaluated node by node, so a memo
	// that could add it but not withdraw it would be invisible to a comparison of states alone.
	existingFlag := func(pl *InterPodAffinity, pod *v1.Pod) bool {
		t.Helper()
		walked, err := antiAffinityNodeList(snapshot, fastPath)
		if err != nil {
			t.Fatalf("anti-affinity node list: %v", err)
		}
		nsLabels := GetNamespaceLabelsSnapshot(klog.FromContext(ctx), pod.Namespace, pl.nsLister)
		_, flag := pl.existingAntiAffinityCounts(ctx, pod, nsLabels, walked, pl.filteringMemoReady())
		return flag
	}

	// Both plugins are driven through the real PreFilter, so what is compared is what Filter would read.
	step := func(pod *v1.Pod, name string) {
		t.Helper()
		want := runPreFilter(t, ctx, fullPlugin, pod, snapshot)
		got := runPreFilter(t, ctx, memoPlugin, pod, snapshot)
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("%s: the memo's counts differ from the full computation (-want,+got):\n%s", name, diff)
		}
		if want.Skipped {
			t.Fatalf("%s: PreFilter Skipped, so the step proves nothing about the memo", name)
		}
		if wantFlag, gotFlag := existingFlag(fullPlugin, pod), existingFlag(memoPlugin, pod); wantFlag != gotFlag {
			t.Errorf("%s: the memo reports host scoped required anti-affinity = %v, the full computation says %v",
				name, gotFlag, wantFlag)
		}
	}

	step(incoming, "initial")

	// Both walked nodes have to be stored, whichever way the hostname fast path classifies their terms.
	// With the gate on, node2 is in the list but its only contribution is the host scoped flag - the zone
	// scoped term on it selects nothing this pod has - and a memo that stored only nodes with counts would
	// drop it, after which the watermark rule would keep reading node2 as "contributes nothing" and the
	// flag would be lost for as long as that node stays as it is.
	if got := onlyExistingEntry(t, memoPlugin, incoming, labels.Set{"team": "infra"}).memo.Len(); got != 2 {
		t.Errorf("the existing counts memo holds %d node contributions, want 2 (node1 and node2)", got)
	}

	// A pod the incoming pod's anti-affinity term selects joins node3.
	added := st.MakePod().Namespace("ns1").Name("added").UID("added").Label("app", "anti").Node("node3").Obj()
	addPod(added)
	step(incoming, "a matching pod added")
	removePod(added)
	step(incoming, "and removed again")

	// A pod that matches nothing joins: no count may move.
	unmatched := st.MakePod().Namespace("ns1").Name("unmatched").UID("unmatched").Label("app", "other").Node("node3").Obj()
	addPod(unmatched)
	step(incoming, "a pod that matches nothing added")
	removePod(unmatched)

	// node3 moves to another zone, which changes every count keyed by zone, and moves back.
	relabelNode("node3", "z3")
	step(incoming, "node3 moved to z3")
	relabelNode("node3", "z2")
	step(incoming, "node3 moved back to z2")

	// The same pod shape in a namespace with different labels is a different key: the namespace labels
	// are what a namespaceSelector term is matched against, and relabeling a namespace moves no
	// generation.
	otherNamespace := filteringMemoIncoming("ns2", "incoming-ns2", nil)
	step(otherNamespace, "the same shape in another namespace")
	step(incoming, "back to the first namespace")

	// And the same shape in a namespace whose labels are identical to the first one's, where nothing but
	// the namespace can separate the two keys. The counts really do differ: the existing pods'
	// anti-affinity terms carry their own namespace, so a pod from ns3 matches none of them.
	sameLabels := filteringMemoIncoming("ns3", "incoming-ns3", nil)
	if diff := cmp.Diff(runPreFilter(t, ctx, fullPlugin, incoming, snapshot),
		runPreFilter(t, ctx, fullPlugin, sameLabels, snapshot)); diff == "" {
		t.Fatalf("ns1 and ns3 count identically, so the next step could not tell a shared key from a separate one")
	}
	step(sameLabels, "the same shape in a namespace with the same labels")
	step(incoming, "back to the first namespace again")

	// The pod that puts the host scoped flag up leaves the cluster, and the memo has to withdraw it -
	// which an aggregate that only ever added could not.
	hostScoped := st.MakePod().Namespace("ns1").Name("anti-host-on-node2").UID("anti-host-on-node2").
		Label("owner", "yes").Node("node2").
		PodAntiAffinityExists("app", v1.LabelHostname, st.PodAntiAffinityWithRequiredReq).Obj()
	removePod(hostScoped)
	step(incoming, "the host scoped anti-affinity pod is gone")
	if existingFlag(fullPlugin, incoming) {
		t.Fatalf("the full computation still reports a host scoped term after the only pod carrying one was removed, so the step proves nothing")
	}
	addPod(hostScoped)
	step(incoming, "and is back")

	// A pod with no terms of its own is not Skipped here, because the existing pods' anti-affinity
	// counts are non-empty, and its own two count maps have to come back empty rather than being walked
	// for.
	before := memoPlugin.filteringIncomingMemo.Len()
	termless := st.MakePod().Namespace("ns1").Name("termless").UID("termless").Label("app", "victim").Obj()
	got := runPreFilter(t, ctx, memoPlugin, termless, snapshot)
	want := runPreFilter(t, ctx, fullPlugin, termless, snapshot)
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("a pod with no terms: (-want,+got):\n%s", diff)
	}
	if got.Skipped {
		t.Errorf("a pod with no terms was Skipped even though existing pods have required anti-affinity against it")
	}
	if len(got.ClusterWideAffinity) != 0 || len(got.ClusterWideAntiAffinity) != 0 {
		t.Errorf("a pod with no terms counted %v / %v, want nothing", got.ClusterWideAffinity, got.ClusterWideAntiAffinity)
	}
	// And it must not have cost a memo entry: there is nothing to memoize, and an entry for a key whose
	// aggregate is empty by construction would only evict a shape that does have something.
	if now := memoPlugin.filteringIncomingMemo.Len(); now != before {
		t.Errorf("a pod with no terms left %d entries in the incoming memo, want the %d it started with", now, before)
	}
}

// TestFilteringMemoReusesUnchangedNodes is the other half: nothing is recomputed while the snapshot
// does not move, and exactly one node is recomputed after one node moves. Without it a memo that
// silently recomputes everything would still pass the comparison above.
func TestFilteringMemoReusesUnchangedNodes(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	snapshot, _ := filteringMemoFixture()
	namespaces := []runtime.Object{
		&v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: "ns1", Labels: map[string]string{"team": "infra"}}},
	}
	memoPlugin, _ := newFilteringMemoPlugin(t, ctx, snapshot, false, namespaces)
	incoming := filteringMemoIncoming("ns1", "incoming", nil)

	runPreFilter(t, ctx, memoPlugin, incoming, snapshot)
	entry, ok := memoPlugin.filteringExistingMemo.Get(existingCountsKey(incoming, map[string]string{"team": "infra"}))
	if !ok {
		t.Fatalf("no memo entry for the incoming pod's shape; keys are %v", memoPlugin.filteringExistingMemo.Keys())
	}
	reused, recomputed := entry.memo.Reused, entry.memo.Recomputed
	if recomputed == 0 {
		t.Fatalf("the first pass recomputed nothing, so there is nothing to reuse either")
	}

	// The same shape over an unmoved snapshot: every node is reused, nothing recomputed.
	runPreFilter(t, ctx, memoPlugin, incoming, snapshot)
	if entry.memo.Recomputed != recomputed {
		t.Errorf("an unmoved snapshot recomputed %d nodes, want 0", entry.memo.Recomputed-recomputed)
	}
	if entry.memo.Reused <= reused {
		t.Errorf("an unmoved snapshot reused %d nodes, want every node of the list", entry.memo.Reused-reused)
	}

	// One node moves: exactly one node is recomputed, in both memos. node1 is the one to move, because
	// it is the node that hosts a pod with required anti-affinity and so is the only fixture node in the
	// filtered list the existing counts walk.
	existingBefore, incomingBefore := entry.memo.Recomputed, incomingRecomputed(t, memoPlugin)
	nodeInfo, err := snapshot.NodeInfos().Get("node1")
	if err != nil {
		t.Fatalf("Get(node1): %v", err)
	}
	moved := nodeInfo.Node().DeepCopy()
	moved.Labels["zone"] = "z3"
	nodeInfo.SetNode(moved)
	runPreFilter(t, ctx, memoPlugin, incoming, snapshot)
	if got := entry.memo.Recomputed - existingBefore; got != 1 {
		t.Errorf("one moved node recomputed %d nodes of the existing counts, want 1", got)
	}
	if got := incomingRecomputed(t, memoPlugin) - incomingBefore; got != 1 {
		t.Errorf("one moved node recomputed %d nodes of the incoming counts, want 1", got)
	}
}

// onlyExistingEntry returns the memo entry the incoming pod's shape is stored under.
func onlyExistingEntry(t *testing.T, pl *InterPodAffinity, pod *v1.Pod, nsLabels labels.Set) *existingCountsEntry {
	t.Helper()
	entry, ok := pl.filteringExistingMemo.Get(existingCountsKey(pod, nsLabels))
	if !ok {
		t.Fatalf("no existing counts entry for %s; keys are %v", pod.Name, pl.filteringExistingMemo.Keys())
	}
	return entry
}

// incomingRecomputed sums the recomputed counter over the incoming memo, which holds one entry per set
// of required terms the plugin has seen.
func incomingRecomputed(t *testing.T, pl *InterPodAffinity) int64 {
	t.Helper()
	var total int64
	for _, key := range pl.filteringIncomingMemo.Keys() {
		entry, ok := pl.filteringIncomingMemo.Get(key)
		if !ok {
			t.Fatalf("the incoming memo lost the entry for %q between listing its keys and reading it", key)
		}
		total += entry.memo.Recomputed
	}
	return total
}

// antiAffinityNodeList returns the snapshot list PreFilter walks for the existing pods' counts, which
// is a different one depending on the hostname fast path.
func antiAffinityNodeList(snapshot *cache.Snapshot, fastPath bool) ([]fwk.NodeInfo, error) {
	if fastPath {
		return snapshot.NodeInfos().HavePodsWithRequiredNonHostScopedAntiAffinityList()
	}
	return snapshot.NodeInfos().HavePodsWithRequiredAntiAffinityList()
}

// TestFilteringMemoKeysCoverTheirInputs pins the keys directly. A key that is coarser than the inputs
// is not a cache miss, it is one shape reading another shape's counts, and the walkthrough above cannot
// show it for inputs the fixture does not vary - the labels of a namespace, which only a
// namespaceSelector term is matched against, and the two selectors that render alike while meaning
// opposite things.
func TestFilteringMemoKeysCoverTheirInputs(t *testing.T) {
	nsLabels := labels.Set{"team": "infra"}
	otherNsLabels := labels.Set{"team": "prod"}

	inNS1 := filteringMemoIncoming("ns1", "incoming", nil)
	// Same pod labels, a namespace whose labels are identical to ns1's: nothing but the namespace itself
	// can separate these two keys.
	inNS3 := filteringMemoIncoming("ns3", "incoming", nil)
	extraLabel := filteringMemoIncoming("ns1", "incoming", func(pod *v1.Pod) { pod.Labels["extra"] = "yes" })

	if got, other := existingCountsKey(inNS1, nsLabels), existingCountsKey(inNS3, nsLabels); got == other {
		t.Errorf("two pods that differ only in their namespace share the existing counts key %q, and the existing pods' terms carry their own namespace", got)
	}
	if got, other := existingCountsKey(inNS1, nsLabels), existingCountsKey(extraLabel, nsLabels); got == other {
		t.Errorf("two pods that differ only in their labels share the existing counts key %q", got)
	}
	if got, other := existingCountsKey(inNS1, nsLabels), existingCountsKey(inNS1, otherNsLabels); got == other {
		t.Errorf("relabeling a namespace does not change the existing counts key %q, and no node generation moves when a namespace is relabeled", got)
	}

	termsOf := func(mutate func(*v1.Pod)) ([]fwk.AffinityTerm, []fwk.AffinityTerm) {
		podInfo, err := framework.NewPodInfo(filteringMemoIncoming("ns1", "incoming", mutate))
		if err != nil {
			t.Fatalf("NewPodInfo: %v", err)
		}
		return podInfo.GetRequiredAffinityTerms(), podInfo.GetRequiredAntiAffinityTerms()
	}
	affinity, antiAffinity := termsOf(nil)
	if got, other := incomingCountsKey(affinity, antiAffinity), incomingCountsKey(affinity, nil); got == other {
		t.Errorf("dropping the anti-affinity terms does not change the incoming counts key %q", got)
	}
	if got, other := incomingCountsKey(affinity, antiAffinity), incomingCountsKey(nil, antiAffinity); got == other {
		t.Errorf("dropping the affinity terms does not change the incoming counts key %q", got)
	}

	// A term with no labelSelector at all matches no pod; a term with an empty one matches every pod in
	// its namespaces. metav1.LabelSelectorAsSelector turns them into labels.Nothing and labels.Everything,
	// and Selector.String() renders both as "".
	noSelector := func(pod *v1.Pod) {
		pod.Spec.Affinity.PodAffinity.RequiredDuringSchedulingIgnoredDuringExecution[0].LabelSelector = nil
		pod.Spec.Affinity.PodAntiAffinity.RequiredDuringSchedulingIgnoredDuringExecution[0].LabelSelector = nil
	}
	emptySelector := func(pod *v1.Pod) {
		pod.Spec.Affinity.PodAffinity.RequiredDuringSchedulingIgnoredDuringExecution[0].LabelSelector = &metav1.LabelSelector{}
		pod.Spec.Affinity.PodAntiAffinity.RequiredDuringSchedulingIgnoredDuringExecution[0].LabelSelector = &metav1.LabelSelector{}
	}
	nothingAffinity, nothingAntiAffinity := termsOf(noSelector)
	everythingAffinity, everythingAntiAffinity := termsOf(emptySelector)
	for _, terms := range [][]fwk.AffinityTerm{nothingAffinity, nothingAntiAffinity} {
		if len(terms) != 1 || !labels.MatchesNothing(terms[0].Selector) {
			t.Fatalf("a term with no labelSelector is supposed to be labels.Nothing(), got %+v", terms)
		}
	}
	for _, terms := range [][]fwk.AffinityTerm{everythingAffinity, everythingAntiAffinity} {
		if len(terms) != 1 || labels.MatchesNothing(terms[0].Selector) {
			t.Fatalf("a term with an empty labelSelector is supposed to be labels.Everything(), got %+v", terms)
		}
	}
	if got, other := incomingCountsKey(nothingAffinity, nothingAntiAffinity),
		incomingCountsKey(everythingAffinity, everythingAntiAffinity); got == other {
		t.Errorf("a term that matches no pod and one that matches every pod share the incoming counts key %q", got)
	}
}

// TestPreFilterSkipsTheMemoInAPodGroupCycle is the reason PreFilter looks at
// CycleState.IsPodGroupSchedulingCycle. A pod group cycle assumes every pod of the gang into one
// snapshot in place, and Snapshot.AssumePod puts the old generation back on purpose so that the
// snapshot stays consistent with the cache - which is exactly the one thing the memo cannot detect. The
// later pods of a gang therefore have to count from the snapshot itself, which is also what lets them
// see the earlier ones.
func TestPreFilterSkipsTheMemoInAPodGroupCycle(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	snapshot, _ := filteringMemoFixture()
	namespaces := []runtime.Object{
		&v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: "ns1", Labels: map[string]string{"team": "infra"}}},
	}
	memoPlugin, fullPlugin := newFilteringMemoPlugin(t, ctx, snapshot, false, namespaces)

	gangPod := func(name string) *v1.Pod {
		return filteringMemoIncoming("ns1", name, nil)
	}
	nodeInfoOf := func(name string) fwk.NodeInfo {
		nodeInfo, err := snapshot.NodeInfos().Get(name)
		if err != nil {
			t.Fatalf("Get(%s): %v", name, err)
		}
		return nodeInfo
	}

	// The first pod of the gang is scheduled in a normal cycle, so the memo is used and learns what node3
	// contributes.
	first := gangPod("gang-0")
	runPreFilter(t, ctx, memoPlugin, first, snapshot)
	entry := onlyExistingEntry(t, memoPlugin, first, labels.Set{"team": "infra"})
	passesBefore := entry.memo.Reused + entry.memo.Recomputed

	// The scheduler now assumes a pod the gang's anti-affinity selects onto node3, in place, the way a
	// pod group cycle does.
	assumed := st.MakePod().Namespace("ns1").Name("assumed").UID("assumed").Label("app", "anti").Obj()
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
	if _, status := memoPlugin.PreFilter(ctx, state, second, nodes); !status.IsSuccess() {
		t.Fatalf("PreFilter(gang-1): %v", status)
	}
	if passes := entry.memo.Reused + entry.memo.Recomputed; passes != passesBefore {
		t.Errorf("the pod group cycle consulted the existing counts memo (%d node lookups, want 0)", passes-passesBefore)
	}

	got, err := getPreFilterState(state)
	if err != nil {
		t.Fatalf("getPreFilterState: %v", err)
	}
	want := runPreFilter(t, ctx, fullPlugin, second, snapshot)
	if diff := cmp.Diff(want.ClusterWideAntiAffinity, renderCounts(got.clusterWideAntiAffinityCounts)); diff != "" {
		t.Errorf("the gang's second pod counted anti-affinity differently from the full computation (-want,+got):\n%s", diff)
	}
	pair := topologyPair{key: "zone", value: "z2"}
	if got.clusterWideAntiAffinityCounts[pair] == 0 {
		t.Errorf("the gang's second pod does not see the pod assumed onto node3 in zone z2, so a gang could be placed on top of its own anti-affinity")
	}
	if status := memoPlugin.Filter(ctx, state, second, nodeInfoOf("node3")); status.IsSuccess() {
		t.Errorf("the gang's second pod was accepted on node3, where a pod it has required anti-affinity against is already assumed")
	}
}
