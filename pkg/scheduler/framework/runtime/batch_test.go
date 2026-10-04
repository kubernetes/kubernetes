/*
Copyright 2025 The Kubernetes Authors.

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

package runtime

import (
	"bytes"
	"context"
	"fmt"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	compbasemetrics "k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
)

// nodeInfoLister declares a fwk.NodeInfo type for testing.
type nodeInfoLister []fwk.NodeInfo

// Get returns a fake node object in the fake nodes.
func (nodes nodeInfoLister) Get(nodeName string) (fwk.NodeInfo, error) {
	for _, node := range nodes {
		if node != nil && node.Node().Name == nodeName {
			return node, nil
		}
	}
	return nil, fmt.Errorf("unable to find node: %s", nodeName)
}

// List lists all nodes.
func (nodes nodeInfoLister) List() ([]fwk.NodeInfo, error) {
	return nodes, nil
}

// HavePodsWithAffinityList is supposed to list nodes with at least one pod with affinity. For the fake lister
// we just return everything.
func (nodes nodeInfoLister) HavePodsWithAffinityList() ([]fwk.NodeInfo, error) {
	return nodes, nil
}

// HavePodsWithRequiredAntiAffinityList is supposed to list nodes with at least one pod with
// required anti-affinity. For the fake lister we just return everything.
func (nodes nodeInfoLister) HavePodsWithRequiredAntiAffinityList() ([]fwk.NodeInfo, error) {
	return nodes, nil
}

// HavePodsWithRequiredNonHostScopedAntiAffinityList is supposed to list nodes with at least one pod with
// required non-host-scoped anti-affinity. For the fake lister we just return everything.
func (nodes nodeInfoLister) HavePodsWithRequiredNonHostScopedAntiAffinityList() ([]fwk.NodeInfo, error) {
	return nodes, nil
}

type sharedLister struct {
	nodes nodeInfoLister
}

func (s sharedLister) NodeInfos() fwk.NodeInfoLister {
	return s.nodes
}

type storageInfoListerContract struct{}

func (c *storageInfoListerContract) IsPVCUsedByPods(_ string) bool {
	return false
}

func (s sharedLister) StorageInfos() fwk.StorageInfoLister {
	return &storageInfoListerContract{}
}

func (s sharedLister) PodGroupStates() fwk.PodGroupStateLister {
	return nil
}

func (s sharedLister) PodGroups() fwk.PodGroupLister {
	return nil
}

func (s sharedLister) CompositePodGroupStates() fwk.CompositePodGroupStateLister {
	return nil
}

func (s sharedLister) CompositePodGroups() fwk.CompositePodGroupLister {
	return nil
}

type BatchTestPlugin struct{}

func (pl *BatchTestPlugin) Name() string {
	return "batchTest"
}

const blockingPodPrefix = 'b'

func blockingPodID(suffix string) string {
	return fmt.Sprintf("%c-%s", blockingPodPrefix, suffix)
}

const nonBlockingPodPrefix = 'a'

func nonBlockingPodID(suffix string) string {
	return fmt.Sprintf("%c-%s", nonBlockingPodPrefix, suffix)
}

// Test plugin assumes that each node can hold only one pod whose id begins with blockingPrefix. This allows
// us to construct pods that block future selves or not.
func (pl *BatchTestPlugin) Filter(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeInfo fwk.NodeInfo) *fwk.Status {
	podID := pod.GetUID()
	for _, nodePod := range nodeInfo.GetPods() {
		npid := nodePod.GetPod().GetUID()
		if podID[0] == blockingPodPrefix && npid[0] == blockingPodPrefix {
			return fwk.NewStatus(fwk.Unschedulable, "unsched")
		}
	}
	return nil
}

func newBatchTestPlugin(_ context.Context, injArgs runtime.Object, f fwk.Handle) (fwk.Plugin, error) {
	return &BatchTestPlugin{}, nil
}

func newBatchTestFramework(ctx context.Context, scorePl *configurableScorePlugin, filterPl *configurableFilterPlugin) (framework.Framework, *sharedLister, error) {
	r := Registry{
		"batchTest":     newBatchTestPlugin,
		queueSortPlugin: newQueueSortPlugin,
		bindPlugin:      newBindPlugin,
	}
	plugins := &config.Plugins{}
	profile := config.KubeSchedulerProfile{Plugins: plugins}
	profile.Plugins.QueueSort.Enabled = []config.Plugin{{Name: queueSortPlugin}}
	profile.Plugins.Bind.Enabled = []config.Plugin{{Name: bindPlugin}}
	profile.Plugins.Filter.Enabled = []config.Plugin{{Name: "batchTest"}}
	if filterPl != nil {
		r["configurableFilter"] = func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) { return filterPl, nil }
		profile.Plugins.Filter.Enabled = append(profile.Plugins.Filter.Enabled, config.Plugin{Name: "configurableFilter"})
	}
	if scorePl != nil {
		r["configurableScore"] = func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) { return scorePl, nil }
		profile.Plugins.PreScore.Enabled = []config.Plugin{{Name: "configurableScore"}}
		profile.Plugins.Score.Enabled = []config.Plugin{{Name: "configurableScore", Weight: 1}}
	}

	lister := &sharedLister{nodes: nodeInfoLister{}}

	ret, err := NewFramework(ctx, r, &profile, WithSnapshotSharedLister(lister))

	return ret, lister, err
}

type testSortedScoredNodes struct {
	Nodes []string
}

var _ framework.SortedScoredNodes = &testSortedScoredNodes{}

func (t *testSortedScoredNodes) Pop() fwk.NodePluginScores {
	ret := fwk.NodePluginScores{Name: t.Nodes[0]}
	t.Nodes = t.Nodes[1:]
	return ret
}

func (t *testSortedScoredNodes) Len() int {
	return len(t.Nodes)
}

func (t *testSortedScoredNodes) UnorderedList() []fwk.NodePluginScores {
	result := make([]fwk.NodePluginScores, len(t.Nodes))
	for i, name := range t.Nodes {
		result[i] = fwk.NodePluginScores{Name: name}
	}
	return result
}

func newTestNodes(n []string) *testSortedScoredNodes {
	return &testSortedScoredNodes{Nodes: n}
}

const (
	batchAttemptsHelp        = "Counts of results when we attempt to use batching, by scheduler profile. 'no_hint' covers every pod that asked for a hint and got none, including pods that later fail to schedule. 'hint_used' and 'hint_not_used' are recorded only when a node is selected, and say whether it was the hinted one."
	batchCacheFlushedHelp    = "Counts of batch cache flushes by reason and scheduler profile. A flush discards a usable batch state. 'pod_skipped' means the previous pod was handled by another profile or failed to schedule, 'node_missing' means a node needed to reuse the state is no longer in the snapshot, and the 'filter_error', 'prescore_error', 'rescore_error' and 'normalize_error' reasons mean that phase failed while rescoring."
	storeScheduleResultsHelp = "Counts of the outcomes of storing scheduling results for batching, by scheduler profile. 'stored' means the pod seeded a batch, 'pod_not_batchable' means the pod has no signature, and 'empty_list' means no other feasible nodes were left to cache. Pods that used the hinted node store nothing."
)

func resetBatchMetrics() {
	metrics.BatchAttemptStats.Reset()
	metrics.BatchCacheFlushed.Reset()
	metrics.StoreScheduleResultsTotal.Reset()
	metrics.BatchRescoreAttempts.Reset()
	metrics.BatchRescoreDuration.Reset()
	metrics.GetNodeHintDuration.Reset()
	metrics.StoreScheduleResultsDuration.Reset()
}

// batchCounterText renders the expected exposition text of a counter with a "profile" label and one more label.
func batchCounterText(name, help, profile, label string, want map[string]int) string {
	if len(want) == 0 {
		return ""
	}
	keys := make([]string, 0, len(want))
	for k := range want {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var sb strings.Builder
	fmt.Fprintf(&sb, "# HELP %s [ALPHA] %s\n# TYPE %s counter\n", name, help, name)
	for _, k := range keys {
		fmt.Fprintf(&sb, "%s{profile=%q,%s=%q} %d\n", name, profile, label, k, want[k])
	}
	return sb.String()
}

// assertBatchCounters checks batch_attempts_total, batch_cache_flushed_total and batch_rescore_attempts_total.
func assertBatchCounters(t *testing.T, profile string, attempts, flushes map[string]int, rescoreAttempts int) {
	t.Helper()
	attemptsText := batchCounterText("scheduler_batch_attempts_total", batchAttemptsHelp, profile, "result", attempts)
	if err := testutil.GatherAndCompare(metrics.GetGather(), strings.NewReader(attemptsText), "scheduler_batch_attempts_total"); err != nil {
		t.Errorf("unexpected batch_attempts_total metric output:\n%v", err)
	}
	flushText := batchCounterText("scheduler_batch_cache_flushed_total", batchCacheFlushedHelp, profile, "reason", flushes)
	if err := testutil.GatherAndCompare(metrics.GetGather(), strings.NewReader(flushText), "scheduler_batch_cache_flushed_total"); err != nil {
		t.Errorf("unexpected batch_cache_flushed_total metric output:\n%v", err)
	}
	rescoreText := ""
	if rescoreAttempts > 0 {
		rescoreText = fmt.Sprintf("# HELP scheduler_batch_rescore_attempts_total [ALPHA] Counts of rescore attempts during opportunistic batching, by scheduler profile.\n"+
			"# TYPE scheduler_batch_rescore_attempts_total counter\n"+
			"scheduler_batch_rescore_attempts_total{profile=%q} %d\n", profile, rescoreAttempts)
	}
	if err := testutil.GatherAndCompare(metrics.GetGather(), strings.NewReader(rescoreText), "scheduler_batch_rescore_attempts_total"); err != nil {
		t.Errorf("unexpected batch_rescore_attempts_total metric output:\n%v", err)
	}
}

// assertStoreResults checks store_schedule_results_total, keyed by the "result" label.
func assertStoreResults(t *testing.T, profile string, want map[string]int) {
	t.Helper()
	text := batchCounterText("scheduler_store_schedule_results_total", storeScheduleResultsHelp, profile, "result", want)
	if err := testutil.GatherAndCompare(metrics.GetGather(), strings.NewReader(text), "scheduler_store_schedule_results_total"); err != nil {
		t.Errorf("unexpected store_schedule_results_total metric output:\n%v", err)
	}
}

func assertHistogramCount(t *testing.T, name string, m compbasemetrics.ObserverMetric, want uint64) {
	t.Helper()
	got, err := testutil.GetHistogramMetricCount(m)
	if err != nil {
		t.Errorf("Failed to get %s sampleCount, err: %v", name, err)
		return
	}
	if got != want {
		t.Errorf("%s: expected %d samples, got %d", name, want, got)
	}
}

func TestBatchBasic(t *testing.T) {
	// This test first let OpportunisticBatch handle the first pod, and then see how it behaves with the second pod.
	tests := []struct {
		name                          string
		firstPodID                    string
		firstSig                      string
		firstPodScheduledSuccessfully bool
		// firstChosenNode is supposed to set only if firstPodScheduledSuccessfully is true.
		firstChosenNode string
		// firstOtherNodes is supposed to set only if firstPodScheduledSuccessfully is true.
		firstOtherNodes framework.SortedScoredNodes
		// if it's true, the test case behaves as if the pods are processed during the same PodGroup scheduling cycle.
		sameCycle bool
		// if it's true, the test case behaves as if there is another pod handled by another profile between the first and second pod.
		skipPod                    bool
		secondPodID                string
		secondPodNominatedNodeName string
		secondSig                  string
		secondChosenNode           string
		secondOtherNodes           framework.SortedScoredNodes
		genericWorkloadEnabled     bool
		expectedHint               string
		expectedState              *batchState
		expectedAttempts           map[string]int
		expectedFlushes            map[string]int
		expectedStoreResults       map[string]int
	}{
		{
			name:                          "a second pod with the same signature gets a hint",
			firstPodID:                    blockingPodID("1"),
			firstSig:                      "sig",
			firstChosenNode:               "n3",
			firstOtherNodes:               newTestNodes([]string{"n1"}),
			firstPodScheduledSuccessfully: true,
			secondPodID:                   blockingPodID("2"),
			secondSig:                     "sig",
			secondChosenNode:              "n1",
			expectedHint:                  "n1",
			expectedAttempts:              map[string]int{metrics.BatchAttemptNoHint: 1, metrics.BatchAttemptHintUsed: 1},
			expectedFlushes:               nil,
			expectedStoreResults:          map[string]int{metrics.StoreResultStored: 1},
		},
		{
			name:                          "a second pod with a different signature doesn't get a hint",
			firstPodID:                    nonBlockingPodID("1"),
			firstSig:                      "sig",
			firstChosenNode:               "n3",
			firstOtherNodes:               newTestNodes([]string{"n1"}),
			firstPodScheduledSuccessfully: true,
			secondPodID:                   nonBlockingPodID("2"),
			secondSig:                     "sig2",
			secondChosenNode:              "n1",
			expectedHint:                  "",
			expectedAttempts:              map[string]int{metrics.BatchAttemptNoHint: 2},
			expectedFlushes:               map[string]int{metrics.BatchFlushPodIncompatible: 1},
			expectedStoreResults:          map[string]int{metrics.StoreResultStored: 1, metrics.StoreResultEmptyList: 1},
		},
		{
			name:                          "pod doesn't get hint if previous pod didn't scheduled",
			firstPodID:                    blockingPodID("1"),
			firstSig:                      "sig",
			firstChosenNode:               "n3",
			firstOtherNodes:               newTestNodes([]string{"n1"}),
			firstPodScheduledSuccessfully: false,
			secondPodID:                   blockingPodID("2"),
			secondSig:                     "sig",
			secondChosenNode:              "n1",
			expectedHint:                  "",
			expectedAttempts:              map[string]int{metrics.BatchAttemptNoHint: 2},
			expectedFlushes:               nil,
			expectedStoreResults:          map[string]int{metrics.StoreResultEmptyList: 1},
		},
		{
			name:                          "empty list",
			firstPodID:                    blockingPodID("1"),
			firstSig:                      "sig",
			firstChosenNode:               "n3",
			firstOtherNodes:               newTestNodes([]string{}),
			firstPodScheduledSuccessfully: true,
			secondPodID:                   blockingPodID("2"),
			secondSig:                     "sig",
			secondChosenNode:              "n4",
			expectedHint:                  "",
			expectedAttempts:              map[string]int{metrics.BatchAttemptNoHint: 2},
			expectedFlushes:               nil,
			expectedStoreResults:          map[string]int{metrics.StoreResultEmptyList: 2},
		},
		{
			name:                          "nil list",
			firstPodID:                    blockingPodID("1"),
			firstSig:                      "sig",
			firstChosenNode:               "n3",
			firstOtherNodes:               nil,
			firstPodScheduledSuccessfully: true,
			secondPodID:                   blockingPodID("2"),
			secondSig:                     "sig",
			secondChosenNode:              "n4",
			expectedHint:                  "",
			expectedAttempts:              map[string]int{metrics.BatchAttemptNoHint: 2},
			expectedFlushes:               nil,
			expectedStoreResults:          map[string]int{metrics.StoreResultEmptyList: 2},
		},
		{
			name:                          "pod doesn't get hint because the previous pod is to a different profile",
			firstPodID:                    blockingPodID("1"),
			firstSig:                      "sig",
			firstChosenNode:               "n3",
			firstOtherNodes:               newTestNodes([]string{"n1"}),
			firstPodScheduledSuccessfully: true,
			skipPod:                       true,
			secondPodID:                   blockingPodID("2"),
			secondSig:                     "sig",
			secondChosenNode:              "n1",
			expectedHint:                  "",
			expectedAttempts:              map[string]int{metrics.BatchAttemptNoHint: 2},
			expectedFlushes:               map[string]int{metrics.BatchFlushPodSkipped: 1},
			expectedStoreResults:          map[string]int{metrics.StoreResultStored: 1, metrics.StoreResultEmptyList: 1},
		},
		{
			name:                          "pod doesn't use batch from preceding pod when they are from the same cycle state, but GenericWorkload is disabled",
			firstPodID:                    blockingPodID("1"),
			firstSig:                      "sig",
			firstChosenNode:               "n3",
			firstOtherNodes:               newTestNodes([]string{"n1"}),
			firstPodScheduledSuccessfully: true,
			sameCycle:                     true,
			secondPodID:                   blockingPodID("2"),
			secondSig:                     "sig",
			secondChosenNode:              "n1",
			genericWorkloadEnabled:        false,
			expectedHint:                  "",
			expectedAttempts:              map[string]int{metrics.BatchAttemptNoHint: 2},
			expectedFlushes:               map[string]int{metrics.BatchFlushPodSkipped: 1},
			expectedStoreResults:          map[string]int{metrics.StoreResultStored: 1, metrics.StoreResultEmptyList: 1},
		},
		{
			name:                          "pod uses batch from preceding pod when they are from the same cycle state and GenericWorkload is enabled",
			firstPodID:                    blockingPodID("1"),
			firstSig:                      "sig",
			firstChosenNode:               "n3",
			firstOtherNodes:               newTestNodes([]string{"n1"}),
			firstPodScheduledSuccessfully: true,
			sameCycle:                     true,
			secondPodID:                   blockingPodID("2"),
			secondSig:                     "sig",
			secondChosenNode:              "n1",
			genericWorkloadEnabled:        true,
			expectedHint:                  "n1",
			expectedAttempts:              map[string]int{metrics.BatchAttemptNoHint: 1, metrics.BatchAttemptHintUsed: 1},
			expectedFlushes:               nil,
			expectedStoreResults:          map[string]int{metrics.StoreResultStored: 1},
		},
		{
			name:                          "pod uses batch from preceding pod and leaves remaining batch to next pod",
			firstPodID:                    blockingPodID("1"),
			firstSig:                      "sig",
			firstChosenNode:               "n3",
			firstOtherNodes:               newTestNodes([]string{"n1", "n2"}),
			firstPodScheduledSuccessfully: true,
			secondPodID:                   blockingPodID("2"),
			secondSig:                     "sig",
			secondChosenNode:              "n1",
			expectedHint:                  "n1",
			expectedState: &batchState{
				signature:   []byte("sig"),
				sortedNodes: newTestNodes([]string{"n2"}),
			},
			expectedAttempts:     map[string]int{metrics.BatchAttemptNoHint: 1, metrics.BatchAttemptHintUsed: 1},
			expectedFlushes:      nil,
			expectedStoreResults: map[string]int{metrics.StoreResultStored: 1},
		},
		{
			name:                          "a second pod with a nominated node does not get a hint",
			firstPodID:                    blockingPodID("1"),
			firstSig:                      "sig",
			firstChosenNode:               "n3",
			firstOtherNodes:               newTestNodes([]string{"n1"}),
			firstPodScheduledSuccessfully: true,
			secondPodID:                   blockingPodID("2"),
			secondPodNominatedNodeName:    "n1",
			secondSig:                     "sig",
			secondChosenNode:              "n1",
			expectedHint:                  "",
			expectedAttempts:              map[string]int{metrics.BatchAttemptNoHint: 2},
			expectedFlushes:               map[string]int{metrics.BatchFlushPodNominated: 1},
			expectedStoreResults:          map[string]int{metrics.StoreResultStored: 1, metrics.StoreResultEmptyList: 1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			resetBatchMetrics()
			testFwk, lister, err := newBatchTestFramework(ctx, nil, nil)
			if err != nil {
				t.Fatalf("Failed to create framework for testing: %v", err)
			}

			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "pod",
					UID:  types.UID(tt.firstPodID),
				},
			}

			signature := fwk.PodSignature(tt.firstSig)
			batch := newOpportunisticBatch(testFwk, tt.genericWorkloadEnabled, time.Minute)
			state := framework.NewCycleState()

			// Run the first "pod" through
			hint := batch.GetNodeHint(ctx, pod, signature, state, 1)
			if hint != "" {
				t.Fatalf("Got unexpected hint %s", hint)
			}
			if tt.firstPodScheduledSuccessfully {
				batch.StoreScheduleResults(ctx, []byte(tt.firstSig), hint, tt.firstChosenNode, tt.firstOtherNodes, 1)
			}

			var cycleCount int64 = 2
			if tt.skipPod {
				cycleCount = 3
			} else if tt.sameCycle {
				cycleCount = 1
			}

			// Run the second pod
			pod2 := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "pod2",
					UID:  types.UID(tt.secondPodID),
				},
				Status: v1.PodStatus{
					NominatedNodeName: tt.secondPodNominatedNodeName,
				},
			}

			lastChosenNode := framework.NewNodeInfo(pod)
			lastChosenNode.SetNode(&v1.Node{ObjectMeta: metav1.ObjectMeta{
				Name: tt.firstChosenNode,
				UID:  types.UID(tt.firstChosenNode),
			}})
			lister.nodes = nodeInfoLister{lastChosenNode}

			signature = fwk.PodSignature(tt.secondSig)
			hint = batch.GetNodeHint(ctx, pod2, signature, state, cycleCount)

			if hint != tt.expectedHint {
				t.Fatalf("Got hint '%s' expected '%s' for test '%s'", hint, tt.expectedHint, tt.name)
			}

			batch.StoreScheduleResults(ctx, []byte(tt.secondSig), hint, tt.secondChosenNode, tt.secondOtherNodes, cycleCount)

			batchEmpty := batch.state == nil || batch.state.sortedNodes == nil || batch.state.sortedNodes.Len() == 0
			expectedEmpty := tt.expectedState == nil

			if batchEmpty != expectedEmpty {
				t.Fatalf("Expected empty %t, got empty %t for %s", expectedEmpty, batchEmpty, tt.name)
			}
			if !expectedEmpty {
				if !bytes.Equal(batch.state.signature, []byte(tt.expectedState.signature)) {
					t.Fatalf("Got state signature '%s' expected '%s' for test '%s'", batch.state.signature, tt.expectedState.signature, tt.name)
				}
				nodesDiff := cmp.Diff(tt.expectedState.sortedNodes, batch.state.sortedNodes)
				if nodesDiff != "" {
					t.Fatalf("Diff between sortedNodes (-want,+got):\n%s", nodesDiff)
				}
			}

			profile := testFwk.ProfileName()
			assertBatchCounters(t, profile, tt.expectedAttempts, tt.expectedFlushes, 0)
			assertStoreResults(t, profile, tt.expectedStoreResults)

			wantHinted, wantNoHint := uint64(0), uint64(2)
			if tt.expectedHint != "" {
				wantHinted, wantNoHint = 1, 1
			}
			assertHistogramCount(t, metrics.GetNodeHintDuration.Name, metrics.GetNodeHintDuration.WithLabelValues("hint", profile), wantHinted)
			assertHistogramCount(t, metrics.GetNodeHintDuration.Name, metrics.GetNodeHintDuration.WithLabelValues("no_hint", profile), wantNoHint)

			wantStores := uint64(1)
			if tt.firstPodScheduledSuccessfully {
				wantStores += 1
			}
			assertHistogramCount(t, metrics.StoreScheduleResultsDuration.Name, metrics.StoreScheduleResultsDuration.WithLabelValues(profile), wantStores)
		})
	}
}

// preScoreStateKey is the CycleState key used by configurableScorePlugin.
const preScoreStateKey fwk.StateKey = "configurableScorePlugin/preScoreSeenNodes"

// preScoreSeenNodes records the node names passed to PreScore so NormalizeScore
// can verify it was called with the full candidate set.
type preScoreSeenNodes map[string]struct{}

func (s preScoreSeenNodes) Clone() fwk.StateData {
	clone := make(preScoreSeenNodes, len(s))
	for k := range s {
		clone[k] = struct{}{}
	}
	return clone
}

type configurableFilterPlugin struct {
	filterErr bool
}

func (pl *configurableFilterPlugin) Name() string { return "configurableFilter" }

func (pl *configurableFilterPlugin) Filter(_ context.Context, _ fwk.CycleState, _ *v1.Pod, _ fwk.NodeInfo) *fwk.Status {
	if pl.filterErr {
		return fwk.AsStatus(fmt.Errorf("injected filter error"))
	}
	return nil
}

type configurableScorePlugin struct {
	score       int64
	preScoreErr bool
	scoreErr    bool
	normErr     bool
}

func (pl *configurableScorePlugin) Name() string { return "configurableScore" }

func (pl *configurableScorePlugin) PreScore(_ context.Context, state fwk.CycleState, _ *v1.Pod, nodes []fwk.NodeInfo) *fwk.Status {
	if pl.preScoreErr {
		return fwk.AsStatus(fmt.Errorf("injected prescore error"))
	}
	seen := make(preScoreSeenNodes, len(nodes))
	for _, n := range nodes {
		seen[n.Node().Name] = struct{}{}
	}
	state.Write(preScoreStateKey, seen)
	return nil
}

func (pl *configurableScorePlugin) Score(_ context.Context, _ fwk.CycleState, _ *v1.Pod, _ fwk.NodeInfo) (int64, *fwk.Status) {
	if pl.scoreErr {
		return 0, fwk.AsStatus(fmt.Errorf("injected score error"))
	}
	return pl.score, nil
}

func (pl *configurableScorePlugin) ScoreExtensions() fwk.ScoreExtensions { return pl }

func (pl *configurableScorePlugin) NormalizeScore(_ context.Context, state fwk.CycleState, _ *v1.Pod, scores fwk.NodeScoreList) *fwk.Status {
	if pl.normErr {
		return fwk.AsStatus(fmt.Errorf("injected normalize error"))
	}
	c, err := state.Read(preScoreStateKey)
	if err != nil {
		return fwk.AsStatus(fmt.Errorf("NormalizeScore: missing PreScore state: %w", err))
	}
	seen := c.(preScoreSeenNodes)
	for _, score := range scores {
		if _, ok := seen[score.Name]; !ok {
			return fwk.AsStatus(fmt.Errorf("NormalizeScore called for node %q which was not passed to PreScore",
				score.Name))
		}
	}
	return nil
}

func TestBatchRescore(t *testing.T) {
	const n1CachedScore = 50 // fixed cached score for n1 across all test cases

	tests := []struct {
		name                   string
		score                  int64 // fresh score for n2 after rescore
		filterErr              bool
		preScoreErr            bool
		scoreErr               bool
		normErr                bool
		chosenNodeMissing      bool
		cachedNodeMissing      bool
		expectedHint           string
		expectedRemainingNodes []string
		wantFlushReason        string
		wantRescoreAttempts    int
	}{
		{
			name:                   "rescored node wins",
			score:                  100,
			expectedHint:           "n2",
			expectedRemainingNodes: []string{"n1"},
			wantFlushReason:        "",
			wantRescoreAttempts:    1,
		},
		{
			name:                   "rescored node loses",
			score:                  10,
			expectedHint:           "n1",
			expectedRemainingNodes: []string{"n2"},
			wantFlushReason:        "",
			wantRescoreAttempts:    1,
		},
		{
			name:                "filter error gives no hint",
			score:               100,
			filterErr:           true,
			expectedHint:        "",
			wantFlushReason:     metrics.BatchFlushFilterError,
			wantRescoreAttempts: 0,
		},
		{
			name:                "prescore error gives no hint",
			preScoreErr:         true,
			expectedHint:        "",
			wantFlushReason:     metrics.BatchFlushPreScoreError,
			wantRescoreAttempts: 1,
		},
		{
			name:                "score error gives no hint",
			scoreErr:            true,
			expectedHint:        "",
			wantFlushReason:     metrics.BatchFlushRescoreError,
			wantRescoreAttempts: 1,
		},
		{
			name:                "normalize error gives no hint",
			normErr:             true,
			expectedHint:        "",
			wantFlushReason:     metrics.BatchFlushNormalizeError,
			wantRescoreAttempts: 1,
		},
		{
			name:                "chosen node missing from lister gives no hint",
			chosenNodeMissing:   true,
			expectedHint:        "",
			wantFlushReason:     metrics.BatchFlushNodeMissing,
			wantRescoreAttempts: 0,
		},
		{
			name:                "cached node missing from lister gives no hint",
			score:               100,
			cachedNodeMissing:   true,
			expectedHint:        "",
			wantFlushReason:     metrics.BatchFlushNodeMissing,
			wantRescoreAttempts: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			scorePl := &configurableScorePlugin{
				score:       tt.score,
				preScoreErr: tt.preScoreErr,
				scoreErr:    tt.scoreErr,
				normErr:     tt.normErr,
			}
			filterPl := &configurableFilterPlugin{
				filterErr: tt.filterErr,
			}
			testFwk, lister, err := newBatchTestFramework(ctx, scorePl, filterPl)
			if err != nil {
				t.Fatalf("Failed to create framework: %v", err)
			}

			// First pod: non-blocking, chosen node is n2, n1 left as other candidate.
			pod1 := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1", UID: types.UID(nonBlockingPodID("1"))}}
			sig := fwk.PodSignature("sig")
			batch := newOpportunisticBatch(testFwk, false, time.Minute)

			cachedNodes := framework.NewSortedScoredNodes([]fwk.NodePluginScores{
				{Name: "n1", RawScores: []fwk.PluginScore{{Name: "configurableScore", Score: n1CachedScore}}},
			})
			batch.GetNodeHint(ctx, pod1, sig, framework.NewCycleState(), 1)
			batch.StoreScheduleResults(ctx, []byte("sig"), "", "n2", cachedNodes, 1)

			// n2 (last chosen node) must be in the lister for refreshHintCandidates
			// to proceed. n1 (cached) is omitted when cachedNodeMissing is set to
			// exercise the resolveCachedCandidates abort path.
			if !tt.chosenNodeMissing {
				n2Info := framework.NewNodeInfo(pod1)
				n2Info.SetNode(&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "n2", UID: "n2"}})
				if tt.cachedNodeMissing {
					lister.nodes = nodeInfoLister{n2Info}
				} else {
					n1Info := framework.NewNodeInfo()
					n1Info.SetNode(&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "n1", UID: "n1"}})
					lister.nodes = nodeInfoLister{n1Info, n2Info}
				}
			}

			pod2 := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod2", UID: types.UID(nonBlockingPodID("2"))}}
			resetBatchMetrics()
			hint := batch.GetNodeHint(ctx, pod2, sig, framework.NewCycleState(), 2)

			if hint != tt.expectedHint {
				t.Fatalf("got hint %q, expected %q", hint, tt.expectedHint)
			}

			var wantFlushes, wantAttempts map[string]int
			if tt.wantFlushReason != "" {
				wantFlushes = map[string]int{tt.wantFlushReason: 1}
			}
			if hint == "" {
				wantAttempts = map[string]int{metrics.BatchAttemptNoHint: 1}
			}
			profile := testFwk.ProfileName()
			assertBatchCounters(t, profile, wantAttempts, wantFlushes, tt.wantRescoreAttempts)
			assertHistogramCount(t, metrics.BatchRescoreDuration.Name, metrics.BatchRescoreDuration.WithLabelValues(profile), uint64(tt.wantRescoreAttempts))

			// Every flush in this test discards a usable state, so GetNodeHint must drop it.
			if tt.wantFlushReason != "" && batch.state != nil {
				t.Fatal("expected batch state to be dropped")
			}
			if tt.expectedRemainingNodes != nil {
				if batch.state == nil || batch.state.sortedNodes == nil {
					t.Fatal("expected non-nil batch state after hint")
				}
				if got, want := batch.state.sortedNodes.Len(), len(tt.expectedRemainingNodes); got != want {
					t.Fatalf("remaining node count: got %d, want %d", got, want)
				}
				for i, want := range tt.expectedRemainingNodes {
					if got := batch.state.sortedNodes.Pop().Name; got != want {
						t.Fatalf("remaining node[%d]: got %q, want %q", i, got, want)
					}
				}
			}
		})
	}
}

// TestBatchRescoreChain verifies a three-pod sequence where rescore fires on each
// scheduling cycle and batch state is correctly maintained throughout.
func TestBatchRescoreChain(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	const n1CachedScore = 50
	// n2 rescores to 100, always beating n1's score of 50.
	scorePl := &configurableScorePlugin{score: 100}
	testFwk, lister, err := newBatchTestFramework(ctx, scorePl, nil)
	if err != nil {
		t.Fatalf("Failed to create framework: %v", err)
	}

	sig := fwk.PodSignature("sig")
	batch := newOpportunisticBatch(testFwk, false, time.Minute)

	// Pod1: n2 chosen, n1 is stored as the only other candidate.
	pod1 := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1", UID: types.UID(nonBlockingPodID("1"))}}
	otherNodes := framework.NewSortedScoredNodes([]fwk.NodePluginScores{
		{Name: "n1", RawScores: []fwk.PluginScore{{Name: "configurableScore", Score: n1CachedScore}}},
	})
	batch.GetNodeHint(ctx, pod1, sig, framework.NewCycleState(), 1)
	batch.StoreScheduleResults(ctx, []byte("sig"), "", "n2", otherNodes, 1)

	n1Info := framework.NewNodeInfo()
	n1Info.SetNode(&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "n1", UID: "n1"}})
	n2Info := framework.NewNodeInfo(pod1)
	n2Info.SetNode(&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "n2", UID: "n2"}})
	lister.nodes = nodeInfoLister{n1Info, n2Info}

	resetBatchMetrics()

	// Pod2: rescore fires (n2 still feasible), n2 wins.
	pod2 := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod2", UID: types.UID(nonBlockingPodID("2"))}}
	hint2 := batch.GetNodeHint(ctx, pod2, sig, framework.NewCycleState(), 2)
	if hint2 != "n2" {
		t.Fatalf("pod2: got hint %q, want %q", hint2, "n2")
	}
	batch.StoreScheduleResults(ctx, []byte("sig"), hint2, "n2", nil, 2)

	// Pod3: rescore fires again, n2 wins again.
	// n1 must remain in state for a potential pod4.
	pod3 := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod3", UID: types.UID(nonBlockingPodID("3"))}}
	hint3 := batch.GetNodeHint(ctx, pod3, sig, framework.NewCycleState(), 3)
	if hint3 != "n2" {
		t.Fatalf("pod3: got hint %q, want %q", hint3, "n2")
	}
	if batch.state == nil || batch.state.sortedNodes == nil || batch.state.sortedNodes.Len() != 1 {
		t.Fatal("expected 1 remaining node in state after pod3 hint")
	}
	if got := batch.state.sortedNodes.Pop().Name; got != "n1" {
		t.Fatalf("remaining node after pod3: got %q, want %q", got, "n1")
	}

	profile := testFwk.ProfileName()
	assertBatchCounters(t, profile, map[string]int{metrics.BatchAttemptHintUsed: 1}, nil, 2)
	assertHistogramCount(t, metrics.BatchRescoreDuration.Name, metrics.BatchRescoreDuration.WithLabelValues(profile), 2)
}

// TestBatchMetricsOtherPaths covers the batch_attempts_total, batch_cache_flushed_total and
// store_schedule_results_total series that TestBatchBasic and TestBatchRescore can't reach through the usual flow.
func TestBatchMetricsOtherPaths(t *testing.T) {
	sig := fwk.PodSignature("sig")
	otherNodes := func() framework.SortedScoredNodes { return newTestNodes([]string{"n1"}) }
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod", UID: types.UID(nonBlockingPodID("1"))}}

	tests := []struct {
		name             string
		run              func(ctx context.Context, batch *OpportunisticBatch)
		wantAttempts     map[string]int
		wantFlushes      map[string]int
		wantStoreResults map[string]int
	}{
		{
			name: "hint given but a different node was chosen",
			run: func(ctx context.Context, batch *OpportunisticBatch) {
				batch.StoreScheduleResults(ctx, sig, "n1", "n2", otherNodes(), 1)
			},
			wantAttempts:     map[string]int{metrics.BatchAttemptHintNotUsed: 1},
			wantStoreResults: map[string]int{metrics.StoreResultStored: 1},
		},
		{
			name: "pod without a signature replaces clears a usable state",
			run: func(ctx context.Context, batch *OpportunisticBatch) {
				batch.StoreScheduleResults(ctx, sig, "", "n2", otherNodes(), 1)
				batch.StoreScheduleResults(ctx, nil, "", "n3", otherNodes(), 2)
			},
			wantStoreResults: map[string]int{metrics.StoreResultStored: 1, metrics.StoreResultPodNotBatchable: 1},
		},
		{
			name: "pod with no other nodes replaces a usable state",
			run: func(ctx context.Context, batch *OpportunisticBatch) {
				batch.StoreScheduleResults(ctx, sig, "", "n2", otherNodes(), 1)
				batch.StoreScheduleResults(ctx, sig, "", "n3", nil, 2)
			},
			wantStoreResults: map[string]int{metrics.StoreResultStored: 1, metrics.StoreResultEmptyList: 1},
		},
		{
			name: "pods that can't seed a batch are counted with no state",
			run: func(ctx context.Context, batch *OpportunisticBatch) {
				batch.StoreScheduleResults(ctx, nil, "", "n3", nil, 1)
				batch.StoreScheduleResults(ctx, sig, "", "n3", nil, 2)
			},
			wantStoreResults: map[string]int{metrics.StoreResultPodNotBatchable: 1, metrics.StoreResultEmptyList: 1},
		},
		{
			name: "batch state older than maxBatchAge expires",
			run: func(ctx context.Context, batch *OpportunisticBatch) {
				batch.StoreScheduleResults(ctx, sig, "", "n2", otherNodes(), 1)
				batch.state.creationTime = time.Now().Add(-2 * time.Minute)
				batch.GetNodeHint(ctx, pod, sig, framework.NewCycleState(), 2)
			},
			wantAttempts:     map[string]int{metrics.BatchAttemptNoHint: 1},
			wantFlushes:      map[string]int{metrics.BatchFlushExpired: 1},
			wantStoreResults: map[string]int{metrics.StoreResultStored: 1},
		},
		{
			name: "a flush is reported once when the following pods fail to schedule",
			run: func(ctx context.Context, batch *OpportunisticBatch) {
				batch.StoreScheduleResults(ctx, sig, "", "n2", otherNodes(), 1)
				for cycle := int64(2); cycle <= 4; cycle++ {
					batch.GetNodeHint(ctx, pod, fwk.PodSignature("other"), framework.NewCycleState(), cycle)
				}
			},
			wantAttempts:     map[string]int{metrics.BatchAttemptNoHint: 3},
			wantFlushes:      map[string]int{metrics.BatchFlushPodIncompatible: 1},
			wantStoreResults: map[string]int{metrics.StoreResultStored: 1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			resetBatchMetrics()

			testFwk, _, err := newBatchTestFramework(ctx, nil, nil)
			if err != nil {
				t.Fatalf("Failed to create framework: %v", err)
			}
			batch := newOpportunisticBatch(testFwk, false, time.Minute)
			tt.run(ctx, batch)

			assertBatchCounters(t, testFwk.ProfileName(), tt.wantAttempts, tt.wantFlushes, 0)
			assertStoreResults(t, testFwk.ProfileName(), tt.wantStoreResults)
		})
	}
}
