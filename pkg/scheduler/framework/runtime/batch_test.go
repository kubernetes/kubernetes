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
	"log"
	"testing"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/framework"
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

var batchRegistry = func() Registry {
	r := make(Registry)
	err := r.Register("batchTest", newBatchTestPlugin)
	if err != nil {
		log.Fatal("Couldn't register test.")
	}
	err = r.Register(queueSortPlugin, newQueueSortPlugin)
	if err != nil {
		log.Fatal("Couldn't register test.")
	}
	err = r.Register(bindPlugin, newBindPlugin)
	if err != nil {
		log.Fatal("Couldn't register test.")
	}
	return r
}()

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

func newBatchTestFramework(ctx context.Context, r Registry) (framework.Framework, *sharedLister, error) {
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

	lister := &sharedLister{nodes: nodeInfoLister{}}

	ret, err := NewFramework(ctx, r, &profile, WithSnapshotSharedLister(lister))

	return ret, lister, err
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
		},
		{
			name:                          "a second pod doesn't get a hint if it's not 1-pod-per-node",
			firstPodID:                    nonBlockingPodID("1"),
			firstSig:                      "sig",
			firstChosenNode:               "n3",
			firstOtherNodes:               newTestNodes([]string{"n1"}),
			firstPodScheduledSuccessfully: true,
			secondPodID:                   nonBlockingPodID("2"),
			secondSig:                     "sig",
			secondChosenNode:              "n1",
			expectedHint:                  "",
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
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			testFwk, lister, err := newBatchTestFramework(ctx, batchRegistry)
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
			batch := newOpportunisticBatch(testFwk, tt.genericWorkloadEnabled)
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
		})
	}
}

// countingFilterPlugin is a filter plugin that records each Filter invocation.
// It also iterates over the node's pods and the pod's containers to simulate
// the non-trivial work that real filter plugins (e.g. NodeResourcesFit) perform.
type countingFilterPlugin struct {
	calls        int
	rejectResult bool // if true, Filter returns Unschedulable
}

func (pl *countingFilterPlugin) Name() string { return "countingFilter" }

func (pl *countingFilterPlugin) Filter(_ context.Context, _ fwk.CycleState, pod *v1.Pod, nodeInfo fwk.NodeInfo) *fwk.Status {
	pl.calls++
	// Simulate work proportional to what a real resource filter does:
	// iterate containers and node pods to avoid the call being compiled away.
	var used int64
	for _, p := range nodeInfo.GetPods() {
		for _, c := range p.GetPod().Spec.Containers {
			if q, ok := c.Resources.Requests[v1.ResourceCPU]; ok {
				used += q.MilliValue()
			}
		}
	}
	for _, c := range pod.Spec.Containers {
		if q, ok := c.Resources.Requests[v1.ResourceCPU]; ok {
			used += q.MilliValue()
		}
	}
	_ = used
	if pl.rejectResult {
		return fwk.NewStatus(fwk.Unschedulable, "always reject")
	}
	return nil
}

func newCountingFilterPlugin(pl *countingFilterPlugin) func(context.Context, runtime.Object, fwk.Handle) (fwk.Plugin, error) {
	return func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
		return pl, nil
	}
}

// makeResourcePod returns a pod with the given CPU and memory requests on its single container.
func makeResourcePod(name string, cpu, mem string) *v1.Pod {
	reqs := v1.ResourceList{}
	if cpu != "" {
		reqs[v1.ResourceCPU] = resource.MustParse(cpu)
	}
	if mem != "" {
		reqs[v1.ResourceMemory] = resource.MustParse(mem)
	}
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name, UID: types.UID(blockingPodID(name))},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{Resources: v1.ResourceRequirements{Requests: reqs}},
			},
		},
	}
}

// makeNodeInfo builds a NodeInfo with the given allocatable and already-requested resources.
func makeNodeInfo(nodeName string, allocMilliCPU, allocMemory, reqMilliCPU, reqMemory int64) *framework.NodeInfo {
	ni := framework.NewNodeInfo()
	ni.SetNode(&v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: nodeName, UID: types.UID(nodeName)},
		Status: v1.NodeStatus{
			Allocatable: v1.ResourceList{
				v1.ResourceCPU:    *resource.NewMilliQuantity(allocMilliCPU, resource.DecimalSI),
				v1.ResourceMemory: *resource.NewQuantity(allocMemory, resource.BinarySI),
			},
		},
	})
	ni.Requested.MilliCPU = reqMilliCPU
	ni.Requested.Memory = reqMemory
	return ni
}

// TestBatchResourcePrecheck verifies that batchStateCompatible skips
// RunFilterPlugins when the pod's resource requests fit within the node's
// remaining allocatable capacity.
func TestBatchResourcePrecheck(t *testing.T) {
	const (
		allocMilliCPU = int64(8000)
		allocMemory   = int64(16 * 1024 * 1024 * 1024)
		reqMilliCPU   = int64(4000) // 4 CPUs already in use
		reqMemory     = int64(8 * 1024 * 1024 * 1024)
	)

	tests := []struct {
		name              string
		podCPU            string
		podMemory         string
		filterShouldPass  bool // filter returns nil (pod fits) when called
		expectFilterCalls int
		expectHint        string
	}{
		{
			name:              "pod fits node resources: filter not called",
			podCPU:            "100m",
			podMemory:         "128Mi",
			filterShouldPass:  true,  // irrelevant; filter won't be called
			expectFilterCalls: 0,
			expectHint:        "",
		},
		{
			name:              "pod exhausts node CPU: filter is called and rejects",
			podCPU:            "5000m", // 5 CPUs but only 4 remaining
			podMemory:         "128Mi",
			filterShouldPass:  false, // filter returns Unschedulable
			expectFilterCalls: 1,
			expectHint:        "n1", // batchStateCompatible returns true → hint given
		},
		{
			name:              "pod exhausts node memory: filter is called and rejects",
			podCPU:            "100m",
			podMemory:         "9Gi", // 9 GiB but only 8 GiB remaining
			filterShouldPass:  false,
			expectFilterCalls: 1,
			expectHint:        "n1",
		},
		{
			name:              "pod has no resource requests: filter is called",
			podCPU:            "",
			podMemory:         "",
			filterShouldPass:  true, // filter passes, so batchStateCompatible returns false
			expectFilterCalls: 1,
			expectHint:        "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			filter := &countingFilterPlugin{rejectResult: !tt.filterShouldPass}
			r := make(Registry)
			if err := r.Register("countingFilter", newCountingFilterPlugin(filter)); err != nil {
				t.Fatal(err)
			}
			if err := r.Register(queueSortPlugin, newQueueSortPlugin); err != nil {
				t.Fatal(err)
			}
			if err := r.Register(bindPlugin, newBindPlugin); err != nil {
				t.Fatal(err)
			}
			plugins := &config.Plugins{
				QueueSort: config.PluginSet{Enabled: []config.Plugin{{Name: queueSortPlugin}}},
				Bind:      config.PluginSet{Enabled: []config.Plugin{{Name: bindPlugin}}},
				Filter:    config.PluginSet{Enabled: []config.Plugin{{Name: "countingFilter"}}},
			}
			profile := config.KubeSchedulerProfile{Plugins: plugins}

			lister := &sharedLister{nodes: nodeInfoLister{}}
			testFwk, err := NewFramework(ctx, r, &profile, WithSnapshotSharedLister(lister))
			if err != nil {
				t.Fatalf("Failed to create framework: %v", err)
			}

			batch := newOpportunisticBatch(testFwk, false)
			state := framework.NewCycleState()

			// Simulate a successfully scheduled first pod that chose "n3".
			firstPod := makeResourcePod("1", tt.podCPU, tt.podMemory)
			firstPod.UID = types.UID(blockingPodID("first"))
			batch.StoreScheduleResults(ctx, fwk.PodSignature("sig"), "", "n3",
				newTestNodes([]string{"n1"}), 1)
			batch.lastCycle = schedulingCycle{cycleCount: 1, chosenNode: "n3", succeeded: true}

			// Set up the node lister with the lastChosenNode ("n3") having known resources.
			nodeN3 := makeNodeInfo("n3", allocMilliCPU, allocMemory, reqMilliCPU, reqMemory)
			lister.nodes = nodeInfoLister{nodeN3}

			// Run the second pod through GetNodeHint.
			secondPod := makeResourcePod("2", tt.podCPU, tt.podMemory)
			hint := batch.GetNodeHint(ctx, secondPod, fwk.PodSignature("sig"), state, 2)

			if hint != tt.expectHint {
				t.Errorf("hint = %q, want %q", hint, tt.expectHint)
			}
			if filter.calls != tt.expectFilterCalls {
				t.Errorf("filter called %d times, want %d", filter.calls, tt.expectFilterCalls)
			}
		})
	}
}

func BenchmarkBatchStateCompatibleResourcePrecheck(b *testing.B) {
	// Benchmark batchStateCompatible for the common case: a low-resource pod
	// on a node with plenty of remaining capacity. The fast path should avoid
	// calling RunFilterPlugins entirely.
	_, ctx := ktesting.NewTestContext(b)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	filter := &countingFilterPlugin{rejectResult: false}
	r := make(Registry)
	_ = r.Register("countingFilter", newCountingFilterPlugin(filter))
	_ = r.Register(queueSortPlugin, newQueueSortPlugin)
	_ = r.Register(bindPlugin, newBindPlugin)
	plugins := &config.Plugins{
		QueueSort: config.PluginSet{Enabled: []config.Plugin{{Name: queueSortPlugin}}},
		Bind:      config.PluginSet{Enabled: []config.Plugin{{Name: bindPlugin}}},
		Filter:    config.PluginSet{Enabled: []config.Plugin{{Name: "countingFilter"}}},
	}
	profile := config.KubeSchedulerProfile{Plugins: plugins}
	lister := &sharedLister{nodes: nodeInfoLister{}}
	testFwk, _ := NewFramework(ctx, r, &profile, WithSnapshotSharedLister(lister))

	nodeN3 := makeNodeInfo("n3", 8000, 16*1024*1024*1024, 4000, 8*1024*1024*1024)
	lister.nodes = nodeInfoLister{nodeN3}

	pod := makeResourcePod("bench", "100m", "128Mi")
	state := framework.NewCycleState()

	batch := newOpportunisticBatch(testFwk, false)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		filter.calls = 0
		batch.state = &batchState{
			signature:   fwk.PodSignature("sig"),
			sortedNodes: newTestNodes([]string{"n1"}),
		}
		batch.lastCycle = schedulingCycle{cycleCount: int64(i), chosenNode: "n3", succeeded: true}
		batch.batchStateCompatible(ctx, pod, fwk.PodSignature("sig"), int64(i+1), state, lister.nodes)
	}
}

// BenchmarkBatchStateCompatibleRunFilterPath measures batchStateCompatible when
// RunFilterPlugins is always reached — equivalent to the pre-optimization path
// for any pod. Use a pod with no resource requests so the fast-path pre-check
// does not fire.
func BenchmarkBatchStateCompatibleRunFilterPath(b *testing.B) {
	_, ctx := ktesting.NewTestContext(b)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	filter := &countingFilterPlugin{rejectResult: false}
	r := make(Registry)
	_ = r.Register("countingFilter", newCountingFilterPlugin(filter))
	_ = r.Register(queueSortPlugin, newQueueSortPlugin)
	_ = r.Register(bindPlugin, newBindPlugin)
	plugins := &config.Plugins{
		QueueSort: config.PluginSet{Enabled: []config.Plugin{{Name: queueSortPlugin}}},
		Bind:      config.PluginSet{Enabled: []config.Plugin{{Name: bindPlugin}}},
		Filter:    config.PluginSet{Enabled: []config.Plugin{{Name: "countingFilter"}}},
	}
	profile := config.KubeSchedulerProfile{Plugins: plugins}
	lister := &sharedLister{nodes: nodeInfoLister{}}
	testFwk, _ := NewFramework(ctx, r, &profile, WithSnapshotSharedLister(lister))

	nodeN3 := makeNodeInfo("n3", 8000, 16*1024*1024*1024, 4000, 8*1024*1024*1024)
	lister.nodes = nodeInfoLister{nodeN3}

	// Pod with no resource requests: fast-path check is bypassed, RunFilterPlugins always runs.
	pod := makeResourcePod("bench-no-res", "", "")
	state := framework.NewCycleState()

	batch := newOpportunisticBatch(testFwk, false)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		filter.calls = 0
		batch.state = &batchState{
			signature:   fwk.PodSignature("sig"),
			sortedNodes: newTestNodes([]string{"n1"}),
		}
		batch.lastCycle = schedulingCycle{cycleCount: int64(i), chosenNode: "n3", succeeded: true}
		batch.batchStateCompatible(ctx, pod, fwk.PodSignature("sig"), int64(i+1), state, lister.nodes)
	}
}

// BenchmarkPodContainerRequests measures the cost of the O(1) resource pre-check
// in isolation, so it can be compared with the cost of a RunFilterPlugins call.
func BenchmarkPodContainerRequests(b *testing.B) {
	pod := makeResourcePod("bench", "100m", "128Mi")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cpu, mem := podContainerRequests(pod)
		_ = cpu
		_ = mem
	}
}

// BenchmarkRunFilterPluginsOnly measures the cost of a single RunFilterPlugins
// call through the framework, for comparison with BenchmarkPodContainerRequests.
func BenchmarkRunFilterPluginsOnly(b *testing.B) {
	_, ctx := ktesting.NewTestContext(b)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	filter := &countingFilterPlugin{rejectResult: false}
	r := make(Registry)
	_ = r.Register("countingFilter", newCountingFilterPlugin(filter))
	_ = r.Register(queueSortPlugin, newQueueSortPlugin)
	_ = r.Register(bindPlugin, newBindPlugin)
	plugins := &config.Plugins{
		QueueSort: config.PluginSet{Enabled: []config.Plugin{{Name: queueSortPlugin}}},
		Bind:      config.PluginSet{Enabled: []config.Plugin{{Name: bindPlugin}}},
		Filter:    config.PluginSet{Enabled: []config.Plugin{{Name: "countingFilter"}}},
	}
	profile := config.KubeSchedulerProfile{Plugins: plugins}
	lister := &sharedLister{nodes: nodeInfoLister{}}
	testFwk, _ := NewFramework(ctx, r, &profile, WithSnapshotSharedLister(lister))

	nodeN3 := makeNodeInfo("n3", 8000, 16*1024*1024*1024, 4000, 8*1024*1024*1024)
	lister.nodes = nodeInfoLister{nodeN3}
	pod := makeResourcePod("bench", "100m", "128Mi")
	state := framework.NewCycleState()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		testFwk.RunFilterPlugins(ctx, state, pod, nodeN3)
	}
}
