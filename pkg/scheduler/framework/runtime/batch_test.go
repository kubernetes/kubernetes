package runtime

import (
	"context"
	"testing"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

var batchRegistry = func() Registry {
	r := make(Registry)
	r.Register("batchTest", newBatchTestPlugin)
	r.Register(queueSortPlugin, newQueueSortPlugin)
	r.Register(bindPlugin, newBindPlugin)
	return r
}()

type BatchTestPlugin struct{}

func (pl *BatchTestPlugin) Name() string {
	return "batchTest"
}

// Test plugin assumes that each node can hold only one node whose id begins with "b". This allows
// us to construct pods that block future selves or not.
func (pl *BatchTestPlugin) Filter(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeInfo fwk.NodeInfo) *fwk.Status {
	podId := pod.GetUID()
	for _, nodePod := range nodeInfo.GetPods() {
		npid := nodePod.GetPod().GetUID()
		if npid[0] == podId[0] && npid[0] == 'b' {
			return fwk.NewStatus(fwk.Unschedulable, "unsched")
		}
	}
	return fwk.NewStatus(fwk.Success, "success")
}

func newBatchTestPlugin(_ context.Context, injArgs runtime.Object, f fwk.Handle) (fwk.Plugin, error) {
	return &BatchTestPlugin{}, nil
}

func newBatchTestFramework(ctx context.Context, r Registry) (framework.Framework, error) {
	plugins := &config.Plugins{}
	profile := config.KubeSchedulerProfile{Plugins: plugins}

	if _, ok := r[queueSortPlugin]; !ok {
		r[queueSortPlugin] = newQueueSortPlugin
	}
	if _, ok := r[bindPlugin]; !ok {
		r[bindPlugin] = newBindPlugin
	}

	if len(profile.Plugins.QueueSort.Enabled) == 0 {
		profile.Plugins.QueueSort.Enabled = append(profile.Plugins.QueueSort.Enabled, config.Plugin{Name: queueSortPlugin})
	}
	if len(profile.Plugins.Bind.Enabled) == 0 {
		profile.Plugins.Bind.Enabled = append(profile.Plugins.Bind.Enabled, config.Plugin{Name: bindPlugin})
	}
	profile.Plugins.Filter.Enabled = []config.Plugin{{Name: "batchTest"}}
	return NewFramework(ctx, r, &profile)
}

type testSortedScoredNodes struct {
	Nodes []string
}

var _ framework.SortedScoredNodes = &testSortedScoredNodes{}

func (t *testSortedScoredNodes) Pop() string {
	ret := t.Nodes[0]
	t.Nodes = t.Nodes[1:]
	return ret
}

func (t *testSortedScoredNodes) Len() int {
	return len(t.Nodes)
}

func newTestNodes(n []string) *testSortedScoredNodes {
	return &testSortedScoredNodes{Nodes: n}
}

func newTestSigFunc(sig *string) PodSignatureFunc {
	return func(p *v1.Pod) string {
		return *sig
	}
}

func TestBatchBasic(t *testing.T) {
	tests := []struct {
		name              string
		firstId           string
		firstSig          string
		firstChosenNode   string
		firstOtherNodes   framework.SortedScoredNodes
		firstPodCompleted bool
		secondId          string
		secondSig         string
		secondChosenNode  string
		secondOtherNodes  framework.SortedScoredNodes
		expectedHint      string
		expectedState     *batchState
	}{
		{
			name:              "single match",
			firstId:           "b1",
			firstSig:          "sig",
			firstChosenNode:   "n3",
			firstOtherNodes:   newTestNodes([]string{"n1"}),
			firstPodCompleted: true,
			secondId:          "b2",
			secondSig:         "sig",
			secondChosenNode:  "n1",
			expectedHint:      "n1",
		},
		{
			name:              "diff sigs",
			firstId:           "a1",
			firstSig:          "sig",
			firstChosenNode:   "n3",
			firstOtherNodes:   newTestNodes([]string{"n1"}),
			firstPodCompleted: true,
			secondId:          "a2",
			secondSig:         "sig2",
			secondChosenNode:  "n1",
			expectedHint:      "",
		},
		{
			name:              "node not filtered",
			firstId:           "a1",
			firstSig:          "sig",
			firstChosenNode:   "n3",
			firstOtherNodes:   newTestNodes([]string{"n1"}),
			firstPodCompleted: true,
			secondId:          "a2",
			secondSig:         "sig",
			secondChosenNode:  "n1",
			expectedHint:      "",
		},
		{
			name:              "incomplete",
			firstId:           "b1",
			firstSig:          "sig",
			firstChosenNode:   "n3",
			firstOtherNodes:   newTestNodes([]string{"n1"}),
			firstPodCompleted: false,
			secondId:          "b2",
			secondSig:         "sig",
			secondChosenNode:  "n1",
			expectedHint:      "",
		},
		{
			name:              "empty list",
			firstId:           "b1",
			firstSig:          "sig",
			firstChosenNode:   "n3",
			firstOtherNodes:   newTestNodes([]string{}),
			firstPodCompleted: true,
			secondId:          "b2",
			secondSig:         "sig",
			secondChosenNode:  "n4",
			expectedHint:      "",
		},
		{
			name:              "nil list",
			firstId:           "b1",
			firstSig:          "sig",
			firstChosenNode:   "n3",
			firstOtherNodes:   nil,
			firstPodCompleted: true,
			secondId:          "b2",
			secondSig:         "sig",
			secondChosenNode:  "n4",
			expectedHint:      "",
		},
		{
			name:              "match multi",
			firstId:           "b1",
			firstSig:          "sig",
			firstChosenNode:   "n3",
			firstOtherNodes:   newTestNodes([]string{"n1", "n2"}),
			firstPodCompleted: true,
			secondId:          "b2",
			secondSig:         "sig",
			secondChosenNode:  "n1",
			expectedHint:      "n1",
			expectedState: &batchState{
				signature:   "sig",
				sortedNodes: newTestNodes([]string{"n2"}),
			},
		},
		{
			name:              "match multi diff choice",
			firstId:           "b1",
			firstSig:          "sig",
			firstChosenNode:   "n3",
			firstOtherNodes:   newTestNodes([]string{"n1", "n2"}),
			firstPodCompleted: true,
			secondId:          "b2",
			secondSig:         "sig",
			secondChosenNode:  "n7",
			expectedHint:      "n1",
		},
	}

	for _, tt := range tests {
		_, ctx := ktesting.NewTestContext(t)
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		testFwk, err := newBatchTestFramework(ctx, batchRegistry)
		if err != nil {
			t.Fatalf("Failed to create framework for testing: %v", err)
		}

		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "pod",
				UID:  types.UID(tt.firstId),
			},
		}

		signature := tt.firstSig
		batch := newOpportunisticBatch(testFwk, newTestSigFunc(&signature))
		state := framework.NewCycleState()

		// Run the first "pod" through
		hint := batch.RunNodeHint(ctx, pod, state, nil)
		if hint != "" {
			t.Fatalf("Got unexpected hint %s", hint)
		}
		batch.StoreScheduleResults(ctx, pod, tt.firstChosenNode, tt.firstOtherNodes)

		if tt.firstPodCompleted {
			batch.PostScore(ctx, true, pod)
		}

		// Run the second pod
		pod2 := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "pod2",
				UID:  types.UID(tt.secondId),
			},
		}

		var lastChosenNode fwk.NodeInfo
		if batch.LastChosen() != "" {
			lastChosenNode = framework.NewNodeInfo(pod)
			lastChosenNode.SetNode(&v1.Node{ObjectMeta: metav1.ObjectMeta{
				Name: batch.LastChosen(),
				UID:  types.UID(batch.LastChosen()),
			}})
		}

		signature = tt.secondSig
		hint = batch.RunNodeHint(ctx, pod2, state, lastChosenNode)

		if hint != tt.expectedHint {
			t.Fatalf("Got hint '%s' expected '%s' for test '%s'", hint, tt.expectedHint, tt.name)
		}

		batch.StoreScheduleResults(ctx, pod, tt.secondChosenNode, tt.secondOtherNodes)

		if tt.firstPodCompleted {
			batch.PostScore(ctx, true, pod2)
		}

		batchEmpty := batch.state == nil || batch.state.sortedNodes == nil || batch.state.sortedNodes.Len() == 0
		expectedEmpty := tt.expectedState == nil

		if batchEmpty != expectedEmpty {
			t.Fatalf("Expected empty %t, got empty %t for %s", expectedEmpty, batchEmpty, tt.name)
		}
		if !expectedEmpty {
			if batch.state.signature != tt.expectedState.signature {
				t.Fatalf("Got state signature '%s' expected '%s' for test '%s'", batch.state.signature, tt.expectedState.signature, tt.name)
			}
			nodesDiff := cmp.Diff(tt.expectedState.sortedNodes, batch.state.sortedNodes)
			if nodesDiff != "" {
				t.Fatalf("Diff between sortedNodes (-want,+got):\n%s", nodesDiff)
			}
		}
	}
}
