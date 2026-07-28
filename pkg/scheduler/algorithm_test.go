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

package scheduler

import (
	"context"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/apis/config"
	internalcache "k8s.io/kubernetes/pkg/scheduler/backend/cache"
	internalqueue "k8s.io/kubernetes/pkg/scheduler/backend/queue"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultbinder"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/queuesort"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	tf "k8s.io/kubernetes/pkg/scheduler/testing/framework"
)

// assumeReserveFixture wires up the pieces every assume/reserve test needs: a cache
// and snapshot holding one node, a framework whose Filter and Reserve verdicts the
// test controls, and the algorithm under test.
type assumeReserveFixture struct {
	algorithm *SchedulingAlgorithm

	fwk framework.Framework
}

func newAssumeReserveFixture(t *testing.T, ctx context.Context, node *v1.Node,
	plugin *assumeReserveTestPlugin, pods ...*v1.Pod) *assumeReserveFixture {
	t.Helper()

	logger := klog.FromContext(ctx)

	objects := []runtime.Object{node}
	queued := make([]runtime.Object, 0, len(pods))
	for _, p := range pods {
		objects = append(objects, p)
		queued = append(queued, p)
	}
	client := clientsetfake.NewClientset(objects...)
	informerFactory := informers.NewSharedInformerFactory(client, 0)
	cache := internalcache.New(ctx, nil, true, false)
	cache.AddNode(logger, node)
	snapshot := internalcache.NewEmptySnapshot()
	queue := internalqueue.NewTestQueueWithObjects(ctx, nil, queued)

	registry := frameworkruntime.Registry{
		queuesort.Name:     queuesort.New,
		defaultbinder.Name: defaultbinder.New,
		"AssumeReserveTestPlugin": func(ctx context.Context, obj runtime.Object, handle fwk.Handle) (fwk.Plugin, error) {
			return plugin, nil
		},
	}
	profileCfg := schedulerapi.KubeSchedulerProfile{
		SchedulerName: "default-scheduler",
		Plugins: &schedulerapi.Plugins{
			QueueSort: schedulerapi.PluginSet{Enabled: []schedulerapi.Plugin{{Name: queuesort.Name}}},
			Filter:    schedulerapi.PluginSet{Enabled: []schedulerapi.Plugin{{Name: "AssumeReserveTestPlugin"}}},
			Reserve:   schedulerapi.PluginSet{Enabled: []schedulerapi.Plugin{{Name: "AssumeReserveTestPlugin"}}},
			Bind:      schedulerapi.PluginSet{Enabled: []schedulerapi.Plugin{{Name: defaultbinder.Name}}},
		},
	}

	schedFwk, err := frameworkruntime.NewFramework(ctx, registry, &profileCfg,
		frameworkruntime.WithInformerFactory(informerFactory),
		frameworkruntime.WithSnapshotSharedLister(snapshot),
		frameworkruntime.WithPodNominator(queue),
	)
	if err != nil {
		t.Fatalf("Failed to create framework: %v", err)
	}
	if err := cache.UpdateSnapshot(logger, snapshot); err != nil {
		t.Fatalf("Failed to update snapshot: %v", err)
	}

	return &assumeReserveFixture{
		algorithm: NewSchedulingAlgorithm(snapshot, cache),
		fwk:       schedFwk,
	}
}

// TestAssumeAndReserve covers the ordinary scheduling cycle, where the placement is
// recorded in the scheduler cache and survives the next snapshot refresh.
func TestAssumeAndReserve(t *testing.T) {
	node1 := st.MakeNode().Name("node1").Obj()
	pod1 := st.MakePod().Name("pod1").Namespace("default").UID("pod1").Obj()

	tests := []struct {
		name               string
		reserveStatus      *fwk.Status
		wantStatusCode     fwk.Code
		wantAssumedInCache bool
	}{
		{
			name:               "success: pod fits on node and is assumed in the cache",
			wantStatusCode:     fwk.Success,
			wantAssumedInCache: true,
		},
		{
			name:               "reserve failure: assumption is rolled back",
			reserveStatus:      fwk.NewStatus(fwk.Error, "reserve fake failure"),
			wantStatusCode:     fwk.Error,
			wantAssumedInCache: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)

			plugin := &assumeReserveTestPlugin{
				fakePodGroupPlugin: &fakePodGroupPlugin{
					filterStatus: map[string]*fwk.Status{pod1.Name: fwk.NewStatus(fwk.Success)},
				},
				reserveStatus: tt.reserveStatus,
			}
			f := newAssumeReserveFixture(t, ctx, node1, plugin, pod1)

			podInfo, err := framework.NewPodInfo(pod1)
			if err != nil {
				t.Fatalf("Failed to create PodInfo: %v", err)
			}
			queuedPodInfo := &framework.QueuedPodInfo{PodInfo: podInfo}

			state := framework.NewCycleState()
			scheduleResult, err := f.algorithm.SchedulePod(ctx, f.fwk, state, queuedPodInfo)
			if err != nil {
				t.Fatalf("schedulePod failed: %v", err)
			}
			if scheduleResult.SuggestedHost != node1.Name {
				t.Fatalf("SuggestedHost = %q, want %q", scheduleResult.SuggestedHost, node1.Name)
			}

			assumedPodInfo, status := f.algorithm.assumeAndReserve(ctx, state, f.fwk, queuedPodInfo, scheduleResult)
			if status.Code() != tt.wantStatusCode {
				t.Errorf("status.Code() = %v, want %v", status.Code(), tt.wantStatusCode)
			}

			if tt.wantStatusCode == fwk.Success {
				if assumedPodInfo.Pod.Spec.NodeName != scheduleResult.SuggestedHost {
					t.Errorf("assumedPodInfo.Pod.Spec.NodeName = %q, want %q", assumedPodInfo.Pod.Spec.NodeName, scheduleResult.SuggestedHost)
				}
				// The queued pod belongs to the caller: assume must work on a copy.
				if pod1.Spec.NodeName != "" {
					t.Errorf("input pod Spec.NodeName mutated = %q, want empty", pod1.Spec.NodeName)
				}
				if assumedPodInfo.Pod == pod1 {
					t.Error("expected assumedPodInfo.Pod to be a deep copy of the input pod, got the same pointer")
				}
			}

			isAssumed, err := f.algorithm.cache.IsAssumedPod(pod1)
			if err != nil {
				t.Fatalf("cache.IsAssumedPod() error: %v", err)
			}
			if isAssumed != tt.wantAssumedInCache {
				t.Errorf("cache.IsAssumedPod() = %v, want %v", isAssumed, tt.wantAssumedInCache)
			}
			// A cache assumption is not visible in the snapshot until it is refreshed.
			if isPodInSnapshot(f.algorithm.nodeInfoSnapshot, node1.Name, pod1.Name) {
				t.Error("pod assumed in the cache should not be in the snapshot yet")
			}

			if tt.wantStatusCode == fwk.Success {
				if plugin.unreserved {
					t.Error("expected Unreserve not to be called yet")
				}
				if err := f.algorithm.unreserveAndForget(ctx, state, f.fwk, assumedPodInfo, scheduleResult.SuggestedHost); err != nil {
					t.Errorf("unreserveAndForget error: %v", err)
				}
				isAssumed, err = f.algorithm.cache.IsAssumedPod(pod1)
				if err != nil {
					t.Fatalf("cache.IsAssumedPod() error after unreserve: %v", err)
				}
				if isAssumed {
					t.Error("pod still assumed in the cache after unreserveAndForget")
				}
				if !plugin.unreserved {
					t.Error("expected Unreserve to be called")
				}
			} else if !plugin.unreserved {
				t.Error("expected Unreserve to be called on reserve failure")
			}
		})
	}
}

// TestAssumeAndReserveWithRevert covers the pod group scheduling cycle, where the
// placement is only tentative: it goes into the snapshot rather than the cache, and
// the returned revert function has to undo every part of it.
func TestAssumeAndReserveWithRevert(t *testing.T) {
	node1 := st.MakeNode().Name("node1").Obj()
	pod1 := st.MakePod().Name("pod1").Namespace("default").UID("pod1").Obj()
	podWithNomination := st.MakePod().Name("pod-nominated").Namespace("default").UID("pod-nominated").NominatedNodeName("node1").Obj()

	tests := []struct {
		name string
		pod  *v1.Pod
	}{
		{
			name: "pod group cycle: assumed in the snapshot only",
			pod:  pod1,
		},
		{
			name: "pod group cycle: nomination is dropped on assume and restored on revert",
			pod:  podWithNomination,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)

			plugin := &assumeReserveTestPlugin{
				fakePodGroupPlugin: &fakePodGroupPlugin{
					filterStatus: map[string]*fwk.Status{tt.pod.Name: fwk.NewStatus(fwk.Success)},
				},
			}
			f := newAssumeReserveFixture(t, ctx, node1, plugin, tt.pod)

			podInfo, err := framework.NewPodInfo(tt.pod)
			if err != nil {
				t.Fatalf("Failed to create PodInfo: %v", err)
			}
			queuedPodInfo := &framework.QueuedPodInfo{PodInfo: podInfo}

			state := newPodGroupCycleState()

			nominated := tt.pod.Status.NominatedNodeName
			if nominated != "" {
				f.fwk.AddNominatedPod(logger, podInfo, &fwk.NominatingInfo{
					NominatedNodeName: nominated,
					NominatingMode:    fwk.ModeOverride,
				})
			}

			scheduleResult, err := f.algorithm.SchedulePod(ctx, f.fwk, state, queuedPodInfo)
			if err != nil {
				t.Fatalf("schedulePod failed: %v", err)
			}

			status, revertFn := f.algorithm.assumeAndReserveWithRevert(ctx, state, f.fwk, queuedPodInfo, scheduleResult)
			if !status.IsSuccess() {
				t.Fatalf("assumeAndReserveWithRevert status = %v, want success", status)
			}
			if revertFn == nil {
				t.Fatal("assumeAndReserveWithRevert returned no revert function on success")
			}

			// The pod group cycle keeps its tentative placement out of the shared cache.
			isAssumed, err := f.algorithm.cache.IsAssumedPod(tt.pod)
			if err != nil {
				t.Fatalf("cache.IsAssumedPod() error: %v", err)
			}
			if isAssumed {
				t.Error("pod assumed in the cache during a pod group cycle, want snapshot only")
			}
			if !isPodInSnapshot(f.algorithm.nodeInfoSnapshot, node1.Name, tt.pod.Name) {
				t.Error("pod not assumed in the snapshot during a pod group cycle")
			}
			if tt.pod.Spec.NodeName != "" {
				t.Errorf("input pod Spec.NodeName mutated = %q, want empty", tt.pod.Spec.NodeName)
			}
			if nominated != "" && len(f.fwk.NominatedPodsForNode(nominated)) != 0 {
				t.Error("expected the nomination to be dropped once the pod was assumed")
			}

			revertFn()

			if isPodInSnapshot(f.algorithm.nodeInfoSnapshot, node1.Name, tt.pod.Name) {
				t.Error("pod still in the snapshot after revert")
			}
			if !plugin.unreserved {
				t.Error("expected Unreserve to be called on revert")
			}
			if nominated != "" && len(f.fwk.NominatedPodsForNode(nominated)) == 0 {
				t.Error("expected the nomination to be restored on revert")
			}
		})
	}
}

// TestAssumeAndReserveSnapshotSync pins down the difference that makes the two
// assume paths worth keeping apart: a cache assumption survives the next snapshot
// refresh, a snapshot-only one does not.
func TestAssumeAndReserveSnapshotSync(t *testing.T) {
	node1 := st.MakeNode().Name("node1").Obj()
	pod1 := st.MakePod().Name("pod1").Namespace("default").UID("pod1").Obj()

	tests := []struct {
		name                       string
		state                      *framework.CycleState
		wantInSnapshotBeforeUpdate bool
		wantInSnapshotAfterUpdate  bool
	}{
		{
			name:                       "ordinary cycle: cache assumption is synced to snapshot on refresh",
			state:                      framework.NewCycleState(),
			wantInSnapshotBeforeUpdate: false,
			wantInSnapshotAfterUpdate:  true,
		},
		{
			name:                       "pod group cycle: snapshot-only assumption is overwritten on refresh",
			state:                      newPodGroupCycleState(),
			wantInSnapshotBeforeUpdate: true,
			wantInSnapshotAfterUpdate:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)

			plugin := &assumeReserveTestPlugin{
				fakePodGroupPlugin: &fakePodGroupPlugin{
					filterStatus: map[string]*fwk.Status{
						pod1.Name: fwk.NewStatus(fwk.Success),
					},
				},
			}
			f := newAssumeReserveFixture(t, ctx, node1, plugin, pod1)

			podInfo, err := framework.NewPodInfo(pod1)
			if err != nil {
				t.Fatalf("Failed to create PodInfo for pod1: %v", err)
			}
			queuedPodInfo := &framework.QueuedPodInfo{PodInfo: podInfo}

			state := tt.state

			scheduleResult, err := f.algorithm.SchedulePod(ctx, f.fwk, state, queuedPodInfo)
			if err != nil {
				t.Fatalf("schedulePod failed for pod1: %v", err)
			}

			if _, status := f.algorithm.assumeAndReserve(ctx, state, f.fwk, queuedPodInfo, scheduleResult); !status.IsSuccess() {
				t.Fatalf("assumeAndReserve failed for pod1: %v", status)
			}

			if got := isPodInSnapshot(f.algorithm.nodeInfoSnapshot, node1.Name, pod1.Name); got != tt.wantInSnapshotBeforeUpdate {
				t.Errorf("isPodInSnapshot before UpdateSnapshot = %v, want %v", got, tt.wantInSnapshotBeforeUpdate)
			}

			if err := f.algorithm.cache.UpdateSnapshot(logger, f.algorithm.nodeInfoSnapshot); err != nil {
				t.Fatalf("cache.UpdateSnapshot failed: %v", err)
			}

			if got := isPodInSnapshot(f.algorithm.nodeInfoSnapshot, node1.Name, pod1.Name); got != tt.wantInSnapshotAfterUpdate {
				t.Errorf("isPodInSnapshot after UpdateSnapshot = %v, want %v", got, tt.wantInSnapshotAfterUpdate)
			}
		})
	}
}

// TestSchedulingAlgorithmDriver covers the Scheduler-level wrapper around the
// algorithm: it owns preemption (PostFilter) and the nominating info, which the
// algorithm itself deliberately knows nothing about.
func TestSchedulingAlgorithmDriver(t *testing.T) {
	node1 := st.MakeNode().Name("node1").Obj()
	pod1 := st.MakePod().Name("pod1").Namespace("default").UID("pod1").Obj()

	tests := []struct {
		name                 string
		filterStatus         *fwk.Status
		postFilterStatus     *fwk.Status
		postFilterResult     *fwk.PostFilterResult
		podGroupCycle        bool
		wantStatusCode       fwk.Code
		wantNominatingInfo   *fwk.NominatingInfo
		wantPostFilterCalled bool
	}{
		{
			name:                 "failure: algorithm finds no nodes",
			filterStatus:         fwk.NewStatus(fwk.Unschedulable, "fake failure"),
			postFilterStatus:     fwk.NewStatus(fwk.Unschedulable),
			wantStatusCode:       fwk.Unschedulable,
			wantPostFilterCalled: true,
		},
		{
			name:                 "pod group cycle: per-pod PostFilter is not run (PodGroupPostFilter handles it)",
			filterStatus:         fwk.NewStatus(fwk.Unschedulable, "no fit"),
			postFilterStatus:     fwk.NewStatus(fwk.Success),
			postFilterResult:     &fwk.PostFilterResult{NominatingInfo: &fwk.NominatingInfo{NominatedNodeName: "node1", NominatingMode: fwk.ModeOverride}},
			podGroupCycle:        true,
			wantStatusCode:       fwk.Unschedulable,
			wantPostFilterCalled: false,
		},
		{
			name:             "failure: algorithm fails but PostFilter nominates a node",
			filterStatus:     fwk.NewStatus(fwk.Unschedulable, "no fit"),
			postFilterStatus: fwk.NewStatus(fwk.Success),
			postFilterResult: &fwk.PostFilterResult{
				NominatingInfo: &fwk.NominatingInfo{
					NominatedNodeName: "node1",
					NominatingMode:    fwk.ModeOverride,
				},
			},
			wantStatusCode: fwk.Unschedulable,
			wantNominatingInfo: &fwk.NominatingInfo{
				NominatedNodeName: "node1",
				NominatingMode:    fwk.ModeOverride,
			},
			wantPostFilterCalled: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			client := clientsetfake.NewClientset(node1, pod1)
			informerFactory := informers.NewSharedInformerFactory(client, 0)
			cache := internalcache.New(ctx, nil, true, false)
			cache.AddNode(logger, node1)
			snapshot := internalcache.NewEmptySnapshot()
			queue := internalqueue.NewTestQueueWithObjects(ctx, nil, []runtime.Object{pod1})

			podInfo, err := framework.NewPodInfo(pod1)
			if err != nil {
				t.Fatalf("Failed to create PodInfo: %v", err)
			}
			queuedPodInfo := &framework.QueuedPodInfo{PodInfo: podInfo}

			fakePlugin := &assumeReserveTestPlugin{
				fakePodGroupPlugin: &fakePodGroupPlugin{
					filterStatus:     map[string]*fwk.Status{pod1.Name: tt.filterStatus},
					postFilterStatus: map[string]*fwk.Status{pod1.Name: tt.postFilterStatus},
					postFilterResult: map[string]*fwk.PostFilterResult{pod1.Name: tt.postFilterResult},
				},
			}

			registry := frameworkruntime.Registry{
				queuesort.Name:     queuesort.New,
				defaultbinder.Name: defaultbinder.New,
				"AssumeReserveTestPlugin": func(ctx context.Context, obj runtime.Object, handle fwk.Handle) (fwk.Plugin, error) {
					return fakePlugin, nil
				},
			}
			profileCfg := schedulerapi.KubeSchedulerProfile{
				SchedulerName: "default-scheduler",
				Plugins: &schedulerapi.Plugins{
					QueueSort:  schedulerapi.PluginSet{Enabled: []schedulerapi.Plugin{{Name: queuesort.Name}}},
					Filter:     schedulerapi.PluginSet{Enabled: []schedulerapi.Plugin{{Name: "AssumeReserveTestPlugin"}}},
					PostFilter: schedulerapi.PluginSet{Enabled: []schedulerapi.Plugin{{Name: "AssumeReserveTestPlugin"}}},
					Bind:       schedulerapi.PluginSet{Enabled: []schedulerapi.Plugin{{Name: defaultbinder.Name}}},
				},
			}

			schedFwk, err := frameworkruntime.NewFramework(ctx, registry, &profileCfg,
				frameworkruntime.WithInformerFactory(informerFactory),
				frameworkruntime.WithSnapshotSharedLister(snapshot),
				frameworkruntime.WithPodNominator(queue),
			)
			if err != nil {
				t.Fatalf("Failed to create framework: %v", err)
			}

			if err := cache.UpdateSnapshot(logger, snapshot); err != nil {
				t.Fatalf("Failed to update snapshot: %v", err)
			}

			sched := &Scheduler{
				Cache:            cache,
				nodeInfoSnapshot: snapshot,
			}
			sched.initAlgorithm()
			sched.applyDefaultHandlers()

			state := framework.NewCycleState()
			if tt.podGroupCycle {
				state.SetPodGroupSchedulingCycle(framework.NewCycleState())
			}

			scheduleResult, status := sched.schedulingAlgorithm(ctx, state, schedFwk, queuedPodInfo, time.Now())

			if status.Code() != tt.wantStatusCode {
				t.Errorf("status.Code() = %v, want %v", status.Code(), tt.wantStatusCode)
			}
			if fakePlugin.postFilterCalled != tt.wantPostFilterCalled {
				t.Errorf("postFilterCalled = %v, want %v", fakePlugin.postFilterCalled, tt.wantPostFilterCalled)
			}
			if diff := cmp.Diff(tt.wantNominatingInfo, scheduleResult.nominatingInfo); diff != "" {
				t.Errorf("Unexpected nominatingInfo (-want,+got):\n%s", diff)
			}
		})
	}
}

// TestFindNodesThatPassFiltersBudget covers the early exit: the search stops once it
// has enough feasible nodes, and "enough" collapses to a single node for profiles
// that neither score nor run extender filters, since there is nothing left to choose
// between the candidates.
func TestFindNodesThatPassFiltersBudget(t *testing.T) {
	pod := st.MakePod().Name("pod1").Obj()
	allNodes := []*v1.Node{
		st.MakeNode().Name("node1").Obj(),
		st.MakeNode().Name("node2").Obj(),
		st.MakeNode().Name("node3").Obj(),
		st.MakeNode().Name("node4").Obj(),
		st.MakeNode().Name("node5").Obj(),
	}

	tests := []struct {
		name             string
		withScoring      bool
		withFilterPlugin bool
		wantCount        int
	}{
		{
			name:             "no scoring and no extenders stops at the first feasible node",
			withFilterPlugin: true,
			wantCount:        1,
		},
		{
			name:             "scoring profile keeps every feasible node to choose from",
			withScoring:      true,
			withFilterPlugin: true,
			wantCount:        len(allNodes),
		},
		{
			name:        "profile without filter plugins still honours the budget",
			withScoring: true,
			wantCount:   len(allNodes),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			snapshot := internalcache.NewSnapshot(nil, allNodes)

			registerPlugins := []tf.RegisterPluginFunc{
				tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			}
			if tt.withFilterPlugin {
				registerPlugins = append(registerPlugins, tf.RegisterFilterPlugin("TrueFilter", tf.NewTrueFilterPlugin))
			}
			if tt.withScoring {
				registerPlugins = append(registerPlugins,
					tf.RegisterScorePlugin("FakeScore", tf.NewFakePreScoreAndScorePlugin("FakeScore", 1, nil, nil), 1))
			}
			schedFwk, err := tf.NewFramework(ctx, registerPlugins, "default-scheduler",
				frameworkruntime.WithSnapshotSharedLister(snapshot),
				frameworkruntime.WithPodNominator(internalqueue.NewTestQueue(ctx, nil)),
			)
			if err != nil {
				t.Fatalf("Failed to create framework: %v", err)
			}

			algo := NewSchedulingAlgorithm(snapshot, nil)

			nodes, err := snapshot.ListNodesInPlacement()
			if err != nil {
				t.Fatalf("ListNodesInPlacement returned error: %v", err)
			}
			diagnosis := framework.Diagnosis{
				NodeToStatus: framework.NewDefaultNodeToStatus(),
			}

			got, err := algo.findNodesThatPassFilters(ctx, schedFwk, framework.NewCycleState(), pod, &diagnosis, nodes)
			if err != nil {
				t.Fatalf("findNodesThatPassFilters returned error: %v", err)
			}
			if len(got) != tt.wantCount {
				t.Errorf("len(nodes) = %d, want %d", len(got), tt.wantCount)
			}

			seen := make(sets.Set[string], len(got))
			for _, n := range got {
				name := n.Node().Name
				if seen.Has(name) {
					t.Errorf("duplicate node name in result: %q", name)
				}
				seen.Insert(name)
			}
		})
	}
}

// TestFindNodesThatPassFiltersEmptyNodes covers an empty candidate list, which
// PreFilter can produce by narrowing to nodes that are not in the placement.
//
// Only the filter-plugin path is exercised: a profile with no filter plugins takes
// the shortcut loop, which indexes nodes[(nextStartNodeIndex+i)%len(nodes)] against
// a budget that the n = 1 rule raises above the empty candidate list, and divides by
// zero. That is pre-existing behaviour, unchanged by this commit, so it is not
// covered here.
func TestFindNodesThatPassFiltersEmptyNodes(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	pod := st.MakePod().Name("pod1").Obj()

	snapshot := internalcache.NewSnapshot(nil, nil)
	algo := NewSchedulingAlgorithm(snapshot, nil)

	schedFwk, err := tf.NewFramework(ctx, []tf.RegisterPluginFunc{
		tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
		tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
		tf.RegisterFilterPlugin("TrueFilter", tf.NewTrueFilterPlugin),
	}, "default-scheduler",
		frameworkruntime.WithPodNominator(internalqueue.NewTestQueue(ctx, nil)),
	)
	if err != nil {
		t.Fatalf("Failed to create framework: %v", err)
	}

	diagnosis := framework.Diagnosis{
		NodeToStatus: framework.NewDefaultNodeToStatus(),
	}
	res, err := algo.findNodesThatPassFilters(ctx, schedFwk, framework.NewCycleState(), pod, &diagnosis, nil)
	if err != nil {
		t.Fatalf("findNodesThatPassFilters returned unexpected error: %v", err)
	}
	if len(res) != 0 {
		t.Errorf("expected empty result, got %d nodes", len(res))
	}
}

// TestCycleProvider checks that the cycle the algorithm was built with is the one the
// batch module sees: it keys its cached scoring state by cycle, so a wrong or
// synthetic counter would silently make unrelated pods look consecutive.
func TestCycleProvider(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	featuregatetesting.SetFeatureGateDuringTest(t, feature.DefaultFeatureGate, features.OpportunisticBatching, true)

	allNodes := []*v1.Node{
		st.MakeNode().Name("node1").Obj(),
	}
	snapshot := internalcache.NewSnapshot(nil, allNodes)
	algo := NewSchedulingAlgorithm(snapshot, nil, WithCurrentCycleProvider(func() int64 { return 42 }))

	registerPlugins := []tf.RegisterPluginFunc{
		tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
		tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
		tf.RegisterFilterPlugin("TrueFilter", tf.NewTrueFilterPlugin),
	}
	schedFwk, err := tf.NewFramework(ctx, registerPlugins, "default-scheduler",
		frameworkruntime.WithSnapshotSharedLister(snapshot),
		frameworkruntime.WithPodNominator(internalqueue.NewTestQueue(ctx, nil)),
	)
	if err != nil {
		t.Fatalf("Failed to create framework: %v", err)
	}

	recordingFwk := &cycleRecordingFramework{Framework: schedFwk}

	pod := st.MakePod().Name("pod1").Obj()
	podInfo, err := framework.NewPodInfo(pod)
	if err != nil {
		t.Fatalf("Failed to create PodInfo: %v", err)
	}
	queuedPodInfo := &framework.QueuedPodInfo{PodInfo: podInfo}

	if _, _, _, err := algo.findNodesThatFitPod(ctx, recordingFwk, framework.NewCycleState(), queuedPodInfo); err != nil {
		t.Fatalf("findNodesThatFitPod returned unexpected error: %v", err)
	}
	if !recordingFwk.hintRequested {
		t.Fatal("findNodesThatFitPod did not ask the batch for a node hint")
	}
	if recordingFwk.recordedCycle != 42 {
		t.Errorf("GetNodeHint received cycleCount = %d, want 42", recordingFwk.recordedCycle)
	}
}

type cycleRecordingFramework struct {
	framework.Framework
	recordedCycle int64
	hintRequested bool
}

func (f *cycleRecordingFramework) GetNodeHint(ctx context.Context, pod *v1.Pod, signature fwk.PodSignature, state fwk.CycleState, cycleCount int64) string {
	f.recordedCycle = cycleCount
	f.hintRequested = true
	return f.Framework.GetNodeHint(ctx, pod, signature, state, cycleCount)
}

var _ fwk.ReservePlugin = &assumeReserveTestPlugin{}

// assumeReserveTestPlugin embeds fakePodGroupPlugin so a single mock instance can test both standard cache assumption
// and pod-group snapshot assumption paths without duplicating plugin registration logic across subtests.
type assumeReserveTestPlugin struct {
	*fakePodGroupPlugin
	reserveStatus *fwk.Status
	unreserved    bool
}

func (p *assumeReserveTestPlugin) Reserve(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) *fwk.Status {
	if p.reserveStatus != nil {
		return p.reserveStatus
	}
	return fwk.NewStatus(fwk.Success)
}

func (p *assumeReserveTestPlugin) Unreserve(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) {
	p.unreserved = true
}

func isPodInSnapshot(snapshot *internalcache.Snapshot, nodeName, podName string) bool {
	if nodeInfo, err := snapshot.Get(nodeName); err == nil {
		for _, p := range nodeInfo.GetPods() {
			if p.GetPod().Name == podName {
				return true
			}
		}
	}
	return false
}

func newPodGroupCycleState() *framework.CycleState {
	state := framework.NewCycleState()
	state.SetPodGroupSchedulingCycle(framework.NewCycleState())
	return state
}
