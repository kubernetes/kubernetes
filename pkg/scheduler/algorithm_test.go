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
	"sort"
	"strings"
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

func TestAssumeAndReserveInCache(t *testing.T) {
	node1 := st.MakeNode().Name("node1").Obj()
	pod1 := st.MakePod().Name("pod1").Namespace("default").UID("pod1").Obj()

	tests := []struct {
		name               string
		pod                *v1.Pod
		filterStatus       *fwk.Status
		reserveStatus      *fwk.Status
		withCache          bool
		wantSuccess        bool
		wantSuggestedHost  string
		wantAssumedInCache bool
		wantAssumedInSnap  bool
		wantErrorMessage   string
	}{
		{
			name:               "success: pod fits on node",
			pod:                pod1,
			filterStatus:       fwk.NewStatus(fwk.Success),
			withCache:          true,
			wantSuccess:        true,
			wantSuggestedHost:  "node1",
			wantAssumedInCache: true,
			wantAssumedInSnap:  false,
		},
		{
			name:               "reserve failure: algorithm succeeds but Reserve plugin fails",
			pod:                pod1,
			filterStatus:       fwk.NewStatus(fwk.Success),
			reserveStatus:      fwk.NewStatus(fwk.Error, "reserve fake failure"),
			withCache:          true,
			wantSuccess:        false,
			wantAssumedInCache: false,
			wantAssumedInSnap:  false,
		},
		{
			name:               "failure: standalone algorithm without cache returns non-success status",
			pod:                pod1,
			filterStatus:       fwk.NewStatus(fwk.Success),
			withCache:          false,
			wantSuccess:        false,
			wantAssumedInCache: false,
			wantAssumedInSnap:  false,
			wantErrorMessage:   "built without a cache",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			client := clientsetfake.NewClientset(node1, tt.pod)
			informerFactory := informers.NewSharedInformerFactory(client, 0)
			cache := internalcache.New(ctx, nil, true, false)
			cache.AddNode(logger, node1)
			snapshot := internalcache.NewEmptySnapshot()
			queue := internalqueue.NewTestQueueWithObjects(ctx, nil, []runtime.Object{tt.pod})

			podInfo, err := framework.NewPodInfo(tt.pod)
			if err != nil {
				t.Fatalf("Failed to create PodInfo: %v", err)
			}
			queuedPodInfo := &framework.QueuedPodInfo{PodInfo: podInfo}

			fakePlugin := &assumeReserveTestPlugin{
				fakePodGroupPlugin: &fakePodGroupPlugin{
					filterStatus: map[string]*fwk.Status{tt.pod.Name: tt.filterStatus},
				},
				reserveStatus: tt.reserveStatus,
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

			var algo *SchedulingAlgorithm
			if tt.withCache {
				algo = NewSchedulingAlgorithm(snapshot, cache, zeroCycle)
			} else {
				algo = NewSchedulingAlgorithm(snapshot, nil, zeroCycle)
			}

			state := framework.NewCycleState()

			scheduleResult, err := algo.schedulePod(ctx, schedFwk, state, queuedPodInfo)
			if err != nil {
				t.Fatalf("SchedulePod failed: %v", err)
			}

			assumedPodInfo, status := algo.AssumeAndReserveInCache(ctx, state, schedFwk, queuedPodInfo, scheduleResult)

			if status.IsSuccess() != tt.wantSuccess {
				t.Errorf("status.IsSuccess() = %v, want %v", status.IsSuccess(), tt.wantSuccess)
			}

			if tt.wantErrorMessage != "" {
				if status == nil || !strings.Contains(status.Message(), tt.wantErrorMessage) {
					t.Errorf("status.Message() = %q, want to contain %q", status.Message(), tt.wantErrorMessage)
				}
			}

			if tt.wantSuccess {
				if assumedPodInfo.Pod.Spec.NodeName != scheduleResult.SuggestedHost {
					t.Errorf("assumedPodInfo.Pod.Spec.NodeName = %q, want %q", assumedPodInfo.Pod.Spec.NodeName, scheduleResult.SuggestedHost)
				}
				if tt.pod.Spec.NodeName != "" {
					t.Errorf("input pod Spec.NodeName mutated = %q, want empty", tt.pod.Spec.NodeName)
				}
				if assumedPodInfo.Pod == tt.pod {
					t.Errorf("expected assumedPodInfo.Pod to be a deep copy of input pod, got same pointer")
				}
			}

			isAssumed, err := cache.IsAssumedPod(tt.pod)
			if err != nil {
				t.Fatalf("cache.IsAssumedPod() error: %v", err)
			}
			if isAssumed != tt.wantAssumedInCache {
				t.Errorf("cache.IsAssumedPod() = %v, want %v", isAssumed, tt.wantAssumedInCache)
			}

			inSnap := isPodInSnapshot(snapshot, "node1", tt.pod.Name)
			if inSnap != tt.wantAssumedInSnap {
				t.Errorf("pod in snapshot = %v, want %v", inSnap, tt.wantAssumedInSnap)
			}

			if tt.wantSuccess {
				if err := algo.UnreserveAndForgetFromCache(ctx, state, schedFwk, assumedPodInfo, scheduleResult.SuggestedHost); err != nil {
					t.Errorf("UnreserveAndForgetFromCache error: %v", err)
				}
				isAssumed, err = cache.IsAssumedPod(tt.pod)
				if err != nil {
					t.Fatalf("cache.IsAssumedPod() error after unreserve: %v", err)
				}
				if isAssumed {
					t.Errorf("pod still assumed in cache after UnreserveAndForgetFromCache")
				}
			}
		})
	}
}

func TestAssumeAndReserveInSnapshot(t *testing.T) {
	node1 := st.MakeNode().Name("node1").Obj()
	pod1 := st.MakePod().Name("pod1").Namespace("default").UID("pod1").Obj()
	podWithNomination := st.MakePod().Name("pod-nominated").Namespace("default").UID("pod-nominated").NominatedNodeName("node1").Obj()

	tests := []struct {
		name               string
		pod                *v1.Pod
		filterStatus       *fwk.Status
		reserveStatus      *fwk.Status
		withCache          bool
		wantSuccess        bool
		wantSuggestedHost  string
		wantAssumedInCache bool
		wantAssumedInSnap  bool
	}{
		{
			// Why wantAssumedInSnap is true and wantAssumedInCache is false: pod-group scheduling cycles evaluate
			// candidates against transient snapshot assumptions to prevent incomplete group transactions from polluting shared cache state.
			name:               "pod group cycle: assumed in snapshot only",
			pod:                pod1,
			filterStatus:       fwk.NewStatus(fwk.Success),
			withCache:          false,
			wantSuccess:        true,
			wantSuggestedHost:  "node1",
			wantAssumedInCache: false,
			wantAssumedInSnap:  true,
		},
		{
			name:               "pod group cycle with cache: assumed in snapshot only",
			pod:                pod1,
			filterStatus:       fwk.NewStatus(fwk.Success),
			withCache:          true,
			wantSuccess:        true,
			wantSuggestedHost:  "node1",
			wantAssumedInCache: false,
			wantAssumedInSnap:  true,
		},
		{
			name:               "pod group cycle: pod with nomination is restored on revert",
			pod:                podWithNomination,
			filterStatus:       fwk.NewStatus(fwk.Success),
			withCache:          true,
			wantSuccess:        true,
			wantSuggestedHost:  "node1",
			wantAssumedInCache: false,
			wantAssumedInSnap:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			client := clientsetfake.NewClientset(node1, tt.pod)
			informerFactory := informers.NewSharedInformerFactory(client, 0)
			cache := internalcache.New(ctx, nil, true, false)
			cache.AddNode(logger, node1)
			snapshot := internalcache.NewEmptySnapshot()
			queue := internalqueue.NewTestQueueWithObjects(ctx, nil, []runtime.Object{tt.pod})

			podInfo, err := framework.NewPodInfo(tt.pod)
			if err != nil {
				t.Fatalf("Failed to create PodInfo: %v", err)
			}
			queuedPodInfo := &framework.QueuedPodInfo{PodInfo: podInfo}

			fakePlugin := &assumeReserveTestPlugin{
				fakePodGroupPlugin: &fakePodGroupPlugin{
					filterStatus: map[string]*fwk.Status{tt.pod.Name: tt.filterStatus},
				},
				reserveStatus: tt.reserveStatus,
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

			var algo *SchedulingAlgorithm
			if tt.withCache {
				algo = NewSchedulingAlgorithm(snapshot, cache, zeroCycle)
			} else {
				algo = NewSchedulingAlgorithm(snapshot, nil, zeroCycle)
			}

			state := framework.NewCycleState()
			state.SetPodGroupSchedulingCycle(framework.NewCycleState())

			if tt.pod.Status.NominatedNodeName != "" {
				queue.AddNominatedPod(logger, podInfo, &fwk.NominatingInfo{
					NominatedNodeName: tt.pod.Status.NominatedNodeName,
					NominatingMode:    fwk.ModeOverride,
				})
			}

			scheduleResult, err := algo.schedulePod(ctx, schedFwk, state, queuedPodInfo)
			if err != nil {
				t.Fatalf("SchedulePod failed: %v", err)
			}

			status, revertFn := algo.AssumeAndReserveInSnapshot(ctx, state, schedFwk, queuedPodInfo, scheduleResult)

			if status.IsSuccess() != tt.wantSuccess {
				t.Errorf("status.IsSuccess() = %v, want %v", status.IsSuccess(), tt.wantSuccess)
			}

			if tt.wantSuccess {
				if tt.pod.Spec.NodeName != "" {
					t.Errorf("input pod Spec.NodeName mutated = %q, want empty", tt.pod.Spec.NodeName)
				}
				if tt.pod.Status.NominatedNodeName != "" {
					if len(queue.NominatedPodsForNode(tt.pod.Status.NominatedNodeName)) != 0 {
						t.Errorf("expected nomination to be removed after successful AssumeAndReserveInSnapshot, but found: %v", queue.NominatedPodsForNode(tt.pod.Status.NominatedNodeName))
					}
				}
			}

			isAssumed, err := cache.IsAssumedPod(tt.pod)
			if err != nil {
				t.Fatalf("cache.IsAssumedPod() error: %v", err)
			}
			if isAssumed != tt.wantAssumedInCache {
				t.Errorf("cache.IsAssumedPod() = %v, want %v", isAssumed, tt.wantAssumedInCache)
			}

			inSnap := isPodInSnapshot(snapshot, "node1", tt.pod.Name)
			if inSnap != tt.wantAssumedInSnap {
				t.Errorf("pod in snapshot = %v, want %v", inSnap, tt.wantAssumedInSnap)
			}

			if revertFn != nil {
				revertFn()
				isAssumed, err = cache.IsAssumedPod(tt.pod)
				if err != nil {
					t.Fatalf("cache.IsAssumedPod() error after revert: %v", err)
				}
				if isAssumed {
					t.Errorf("pod still assumed in cache after revert")
				}
				if isPodInSnapshot(snapshot, "node1", tt.pod.Name) {
					t.Errorf("pod still in snapshot after revert")
				}
				if tt.reserveStatus == nil && !fakePlugin.unreserved {
					t.Errorf("expected Unreserve to be called on revert")
				}
				if tt.pod.Status.NominatedNodeName != "" {
					if len(queue.NominatedPodsForNode(tt.pod.Status.NominatedNodeName)) == 0 {
						t.Errorf("expected nomination to be restored after revert, but none found")
					}
				}
			}
		})
	}
}

func TestAssumeAndReserveSnapshotSync(t *testing.T) {
	node1 := st.MakeNode().Name("node1").Obj()
	pod1 := st.MakePod().Name("pod1").Namespace("default").UID("pod1").Obj()
	pod2 := st.MakePod().Name("pod2").Namespace("default").UID("pod2").Obj()

	logger, ctx := ktesting.NewTestContext(t)
	client := clientsetfake.NewClientset(node1, pod1, pod2)
	informerFactory := informers.NewSharedInformerFactory(client, 0)
	cache := internalcache.New(ctx, nil, true, false)
	cache.AddNode(logger, node1)
	snapshot := internalcache.NewEmptySnapshot()
	queue := internalqueue.NewTestQueueWithObjects(ctx, nil, []runtime.Object{pod1, pod2})

	fakePlugin := &assumeReserveTestPlugin{
		fakePodGroupPlugin: &fakePodGroupPlugin{
			filterStatus: map[string]*fwk.Status{
				pod1.Name: fwk.NewStatus(fwk.Success),
				pod2.Name: fwk.NewStatus(fwk.Success),
			},
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

	algo := NewSchedulingAlgorithm(snapshot, cache, zeroCycle)

	// 1. Assume pod1 via AssumeAndReserveInCache
	pod1Info, _ := framework.NewPodInfo(pod1)
	queuedPod1Info := &framework.QueuedPodInfo{PodInfo: pod1Info}
	state1 := framework.NewCycleState()
	res1, err := algo.schedulePod(ctx, schedFwk, state1, queuedPod1Info)
	if err != nil {
		t.Fatalf("SchedulePod failed for pod1: %v", err)
	}
	_, status1 := algo.AssumeAndReserveInCache(ctx, state1, schedFwk, queuedPod1Info, res1)
	if !status1.IsSuccess() {
		t.Fatalf("AssumeAndReserveInCache failed: %v", status1)
	}

	// 2. Assume pod2 via AssumeAndReserveInSnapshot
	pod2Info, _ := framework.NewPodInfo(pod2)
	queuedPod2Info := &framework.QueuedPodInfo{PodInfo: pod2Info}
	state2 := framework.NewCycleState()
	state2.SetPodGroupSchedulingCycle(framework.NewCycleState())
	res2, err := algo.schedulePod(ctx, schedFwk, state2, queuedPod2Info)
	if err != nil {
		t.Fatalf("SchedulePod failed for pod2: %v", err)
	}
	status2, _ := algo.AssumeAndReserveInSnapshot(ctx, state2, schedFwk, queuedPod2Info, res2)
	if !status2.IsSuccess() {
		t.Fatalf("AssumeAndReserveInSnapshot failed: %v", status2)
	}

	// Before UpdateSnapshot: pod2 is in snapshot, pod1 is in cache but not yet in snapshot
	if !isPodInSnapshot(snapshot, "node1", pod2.Name) {
		t.Errorf("expected pod2 to be in snapshot before UpdateSnapshot")
	}

	// UpdateSnapshot: syncs cache to snapshot, clearing snapshot-only assumptions
	if err := cache.UpdateSnapshot(logger, snapshot); err != nil {
		t.Fatalf("cache.UpdateSnapshot failed: %v", err)
	}

	// After UpdateSnapshot: pod1 (assumed in cache) IS visible in snapshot; pod2 (assumed in snapshot only) is GONE
	if !isPodInSnapshot(snapshot, "node1", pod1.Name) {
		t.Errorf("expected pod1 (assumed in cache) to be visible in snapshot after UpdateSnapshot")
	}
	if isPodInSnapshot(snapshot, "node1", pod2.Name) {
		t.Errorf("expected pod2 (assumed in snapshot) to be gone from snapshot after UpdateSnapshot")
	}
}

func TestSchedulingAlgorithmDriver(t *testing.T) {
	node1 := st.MakeNode().Name("node1").Obj()
	pod1 := st.MakePod().Name("pod1").Namespace("default").UID("pod1").Obj()

	tests := []struct {
		name                 string
		pod                  *v1.Pod
		filterStatus         *fwk.Status
		postFilterStatus     *fwk.Status
		postFilterResult     *fwk.PostFilterResult
		podGroupCycle        bool
		wantSuccess          bool
		wantNominatingInfo   *fwk.NominatingInfo
		wantPostFilterCalled bool
	}{
		{
			name:                 "failure: algorithm finds no nodes",
			pod:                  pod1,
			filterStatus:         fwk.NewStatus(fwk.Unschedulable, "fake failure"),
			postFilterStatus:     fwk.NewStatus(fwk.Unschedulable),
			wantSuccess:          false,
			wantPostFilterCalled: true,
		},
		{
			name:                 "pod group cycle: per-pod PostFilter is not run (PodGroupPostFilter handles it)",
			pod:                  pod1,
			filterStatus:         fwk.NewStatus(fwk.Unschedulable, "no fit"),
			postFilterStatus:     fwk.NewStatus(fwk.Success),
			postFilterResult:     &fwk.PostFilterResult{NominatingInfo: &fwk.NominatingInfo{NominatedNodeName: "node1", NominatingMode: fwk.ModeOverride}},
			podGroupCycle:        true,
			wantSuccess:          false,
			wantPostFilterCalled: false,
		},
		{
			name:             "failure: algorithm fails but PostFilter nominates node",
			pod:              pod1,
			filterStatus:     fwk.NewStatus(fwk.Unschedulable, "no fit"),
			postFilterStatus: fwk.NewStatus(fwk.Success),
			postFilterResult: &fwk.PostFilterResult{
				NominatingInfo: &fwk.NominatingInfo{
					NominatedNodeName: "node1",
					NominatingMode:    fwk.ModeOverride,
				},
			},
			wantSuccess: false,
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
			client := clientsetfake.NewClientset(node1, tt.pod)
			informerFactory := informers.NewSharedInformerFactory(client, 0)
			cache := internalcache.New(ctx, nil, true, false)
			cache.AddNode(logger, node1)
			snapshot := internalcache.NewEmptySnapshot()
			queue := internalqueue.NewTestQueueWithObjects(ctx, nil, []runtime.Object{tt.pod})

			podInfo, err := framework.NewPodInfo(tt.pod)
			if err != nil {
				t.Fatalf("Failed to create PodInfo: %v", err)
			}
			queuedPodInfo := &framework.QueuedPodInfo{PodInfo: podInfo}

			fakePlugin := &assumeReserveTestPlugin{
				fakePodGroupPlugin: &fakePodGroupPlugin{
					filterStatus:     map[string]*fwk.Status{tt.pod.Name: tt.filterStatus},
					postFilterStatus: map[string]*fwk.Status{tt.pod.Name: tt.postFilterStatus},
					postFilterResult: map[string]*fwk.PostFilterResult{tt.pod.Name: tt.postFilterResult},
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

			if status.IsSuccess() != tt.wantSuccess {
				t.Errorf("status.IsSuccess() = %v, want %v", status.IsSuccess(), tt.wantSuccess)
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

func TestFindAllNodesThatFitPod(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	allNodes := []*v1.Node{
		st.MakeNode().Name("node1").Obj(),
		st.MakeNode().Name("node2").Obj(),
		st.MakeNode().Name("node3").Obj(),
	}

	tests := []struct {
		name                    string
		nodes                   []*v1.Node
		pod                     *v1.Pod
		registerPlugins         []tf.RegisterPluginFunc
		extenders               []fwk.Extender
		wantNodeNames           []string
		wantUnschedulablePlugin string
	}{
		{
			name:  "N nodes, Filter plugin admitting a named subset",
			nodes: allNodes,
			pod:   st.MakePod().Name("pod1").Obj(),
			registerPlugins: []tf.RegisterPluginFunc{
				tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
				tf.RegisterFilterPlugin("SubsetFilter", tf.NewFakeFilterPlugin(map[string]fwk.Code{
					"node2": fwk.Unschedulable,
				})),
			},
			wantNodeNames: []string{"node1", "node3"},
		},
		{
			name:  "Pod with nominated node name does not take shortcut and returns all feasible nodes",
			nodes: allNodes[:2],
			pod:   st.MakePod().Name("pod1").NominatedNodeName("node1").Obj(),
			registerPlugins: []tf.RegisterPluginFunc{
				tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
				tf.RegisterFilterPlugin("TrueFilter", tf.NewTrueFilterPlugin),
			},
			wantNodeNames: []string{"node1", "node2"},
		},
		{
			name:  "Profile with no score plugins returns all feasible nodes",
			nodes: allNodes,
			pod:   st.MakePod().Name("pod1").Obj(),
			registerPlugins: []tf.RegisterPluginFunc{
				tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
				tf.RegisterFilterPlugin("TrueFilter", tf.NewTrueFilterPlugin),
			},
			wantNodeNames: []string{"node1", "node2", "node3"},
		},
		{
			name:  "PreFilter plugin returning node subset respects narrowing",
			nodes: allNodes,
			pod:   st.MakePod().Name("pod1").Obj(),
			registerPlugins: []tf.RegisterPluginFunc{
				tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
				tf.RegisterPreFilterPlugin("SubsetPreFilter", tf.NewFakePreFilterPlugin("SubsetPreFilter", &fwk.PreFilterResult{
					NodeNames: sets.New("node2", "node3"),
				}, nil)),
				tf.RegisterFilterPlugin("TrueFilter", tf.NewTrueFilterPlugin),
			},
			wantNodeNames: []string{"node2", "node3"},
		},
		{
			name:  "Filter extender rejecting one node excludes node and adds ExtenderName to UnschedulablePlugins",
			nodes: allNodes[:2],
			pod:   st.MakePod().Name("pod1").Obj(),
			registerPlugins: []tf.RegisterPluginFunc{
				tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
				tf.RegisterFilterPlugin("TrueFilter", tf.NewTrueFilterPlugin),
			},
			extenders: []fwk.Extender{
				&tf.FakeExtender{
					ExtenderName: "FakeExtender",
					Predicates:   []tf.FitPredicate{tf.Node2PredicateExtender},
				},
			},
			wantNodeNames:           []string{"node2"},
			wantUnschedulablePlugin: framework.ExtenderName,
		},
		{
			name:  "Empty snapshot with zero nodes returns empty result without error",
			nodes: nil,
			pod:   st.MakePod().Name("pod1").Obj(),
			registerPlugins: []tf.RegisterPluginFunc{
				tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
				tf.RegisterFilterPlugin("TrueFilter", tf.NewTrueFilterPlugin),
			},
			wantNodeNames: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			snapshot := internalcache.NewSnapshot(nil, tt.nodes)
			schedFwk, err := tf.NewFramework(ctx, tt.registerPlugins, "default-scheduler",
				frameworkruntime.WithSnapshotSharedLister(snapshot),
				frameworkruntime.WithPodNominator(internalqueue.NewTestQueue(ctx, nil)),
				frameworkruntime.WithExtenders(tt.extenders),
			)
			if err != nil {
				t.Fatalf("Failed to create framework: %v", err)
			}

			algo := NewSchedulingAlgorithm(snapshot, nil, zeroCycle)
			podInfo, _ := framework.NewPodInfo(tt.pod)
			queuedPodInfo := &framework.QueuedPodInfo{PodInfo: podInfo}

			nodes, diagnosis, err := algo.FindAllNodesThatFitPod(ctx, framework.NewCycleState(), schedFwk, queuedPodInfo)
			if err != nil {
				t.Fatalf("FindAllNodesThatFitPod returned unexpected error: %v", err)
			}

			gotNames := nodeNamesFromNodeInfos(nodes)
			if diff := cmp.Diff(tt.wantNodeNames, gotNames); diff != "" {
				t.Errorf("FindAllNodesThatFitPod() returned unexpected nodes (-want,+got):\n%s", diff)
			}

			if tt.wantUnschedulablePlugin != "" {
				if !diagnosis.UnschedulablePlugins.Has(tt.wantUnschedulablePlugin) {
					t.Errorf("expected %q in UnschedulablePlugins, got %v", tt.wantUnschedulablePlugin, sets.List(diagnosis.UnschedulablePlugins))
				}
			}
		})
	}
}

func TestFindAllNodesThatFitPodIdempotency(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	allNodes := []*v1.Node{
		st.MakeNode().Name("node1").Obj(),
		st.MakeNode().Name("node2").Obj(),
		st.MakeNode().Name("node3").Obj(),
		st.MakeNode().Name("node4").Obj(),
		st.MakeNode().Name("node5").Obj(),
	}

	tests := []struct {
		name    string
		podName string
	}{
		{
			name:    "Consecutive calls on same instance return identical results and preserve zero next start node index",
			podName: "pod-double",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			snapshot := internalcache.NewSnapshot(nil, allNodes)
			registerPlugins := []tf.RegisterPluginFunc{
				tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			}
			schedFwk, err := tf.NewFramework(ctx, registerPlugins, "default-scheduler",
				frameworkruntime.WithSnapshotSharedLister(snapshot),
				frameworkruntime.WithPodNominator(internalqueue.NewTestQueue(ctx, nil)),
			)
			if err != nil {
				t.Fatalf("Failed to create framework: %v", err)
			}

			algo := NewSchedulingAlgorithm(snapshot, nil, zeroCycle)
			pod := st.MakePod().Name(tt.podName).Obj()
			podInfo, err := framework.NewPodInfo(pod)
			if err != nil {
				t.Fatalf("Failed to create PodInfo: %v", err)
			}
			queuedPodInfo := &framework.QueuedPodInfo{PodInfo: podInfo}

			nodes1, _, err := algo.FindAllNodesThatFitPod(ctx, framework.NewCycleState(), schedFwk, queuedPodInfo)
			if err != nil {
				t.Fatalf("first FindAllNodesThatFitPod call failed: %v", err)
			}
			if algo.nextStartNodeIndex != 0 {
				t.Errorf("after first call nextStartNodeIndex = %d, want 0", algo.nextStartNodeIndex)
			}

			nodes2, _, err := algo.FindAllNodesThatFitPod(ctx, framework.NewCycleState(), schedFwk, queuedPodInfo)
			if err != nil {
				t.Fatalf("second FindAllNodesThatFitPod call failed: %v", err)
			}
			if algo.nextStartNodeIndex != 0 {
				t.Errorf("after second call nextStartNodeIndex = %d, want 0", algo.nextStartNodeIndex)
			}

			got1 := nodeNamesInOrder(nodes1)
			got2 := nodeNamesInOrder(nodes2)
			if diff := cmp.Diff(got1, got2); diff != "" {
				t.Errorf("consecutive calls returned different results (-first,+second):\n%s", diff)
			}
		})
	}
}

func TestFindNodesThatPassFiltersBudget(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
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
		numNodesToFind   int32
		withFilterPlugin bool
		wantCount        int
		wantNoDuplicates bool
	}{
		{
			name:             "budget smaller than the node count stops at the budget",
			numNodesToFind:   3,
			withFilterPlugin: true,
			wantCount:        3,
		},
		{
			name:             "budget larger than the node count is clamped and yields no duplicates",
			numNodesToFind:   int32(len(allNodes)) + 10,
			withFilterPlugin: false,
			wantCount:        len(allNodes),
			wantNoDuplicates: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			snapshot := internalcache.NewSnapshot(nil, allNodes)

			registerPlugins := []tf.RegisterPluginFunc{
				tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			}
			if tt.withFilterPlugin {
				registerPlugins = append(registerPlugins, tf.RegisterFilterPlugin("TrueFilter", tf.NewTrueFilterPlugin))
			}
			schedFwk, err := tf.NewFramework(ctx, registerPlugins, "default-scheduler",
				frameworkruntime.WithSnapshotSharedLister(snapshot),
				frameworkruntime.WithPodNominator(internalqueue.NewTestQueue(ctx, nil)),
			)
			if err != nil {
				t.Fatalf("Failed to create framework: %v", err)
			}

			algo := NewSchedulingAlgorithm(snapshot, nil, zeroCycle)

			nodes, err := snapshot.ListNodesInPlacement()
			if err != nil {
				t.Fatalf("ListNodesInPlacement returned error: %v", err)
			}
			diagnosis := framework.Diagnosis{
				NodeToStatus: framework.NewDefaultNodeToStatus(),
			}

			got, err := algo.findNodesThatPassFilters(
				ctx, schedFwk, framework.NewCycleState(), pod, &diagnosis, nodes, tt.numNodesToFind)
			if err != nil {
				t.Fatalf("findNodesThatPassFilters returned error: %v", err)
			}

			if len(got) != tt.wantCount {
				t.Errorf("len(nodes) = %d, want %d", len(got), tt.wantCount)
			}
			if tt.wantNoDuplicates {
				seen := make(map[string]bool, len(got))
				for _, n := range got {
					name := n.Node().Name
					if seen[name] {
						t.Errorf("duplicate node name in result: %q", name)
					}
					seen[name] = true
				}
			}
		})
	}
}

func TestFindNodesThatPassFiltersEmptyNodes(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	pod := st.MakePod().Name("pod1").Obj()

	snapshot := internalcache.NewSnapshot(nil, nil)
	algo := NewSchedulingAlgorithm(snapshot, nil, zeroCycle)

	tests := []struct {
		name        string
		withFilters bool
	}{
		{
			name:        "With filter plugins on empty or nil nodes",
			withFilters: true,
		},
		{
			name:        "Without filter plugins on empty or nil nodes",
			withFilters: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			registerPlugins := []tf.RegisterPluginFunc{
				tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			}
			if tt.withFilters {
				registerPlugins = append(registerPlugins, tf.RegisterFilterPlugin("TrueFilter", tf.NewTrueFilterPlugin))
			}
			schedFwk, err := tf.NewFramework(ctx, registerPlugins, "default-scheduler",
				frameworkruntime.WithPodNominator(internalqueue.NewTestQueue(ctx, nil)),
			)
			if err != nil {
				t.Fatalf("Failed to create framework: %v", err)
			}

			var diagnosis framework.Diagnosis
			res, err := algo.findNodesThatPassFilters(ctx, schedFwk, framework.NewCycleState(), pod, &diagnosis, nil, 1)
			if err != nil {
				t.Fatalf("findNodesThatPassFilters returned unexpected error: %v", err)
			}
			if len(res) != 0 {
				t.Errorf("expected empty result, got %d nodes", len(res))
			}
		})
	}
}

func TestFindAllNodesThatFitPodWithOpportunisticBatching(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	featuregatetesting.SetFeatureGateDuringTest(t, feature.DefaultFeatureGate, features.OpportunisticBatching, true)

	allNodes := []*v1.Node{
		st.MakeNode().Name("node1").Obj(),
		st.MakeNode().Name("node2").Obj(),
	}
	snapshot := internalcache.NewSnapshot(nil, allNodes)
	algo := NewSchedulingAlgorithm(snapshot, nil, zeroCycle)

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

	pod := st.MakePod().Name("pod1").Obj()
	podInfo, err := framework.NewPodInfo(pod)
	if err != nil {
		t.Fatalf("Failed to create PodInfo: %v", err)
	}
	queuedPodInfo := &framework.QueuedPodInfo{PodInfo: podInfo}

	nodes, _, err := algo.FindAllNodesThatFitPod(ctx, framework.NewCycleState(), schedFwk, queuedPodInfo)
	if err != nil {
		t.Fatalf("FindAllNodesThatFitPod returned unexpected error: %v", err)
	}
	if diff := cmp.Diff([]string{"node1", "node2"}, nodeNamesFromNodeInfos(nodes)); diff != "" {
		t.Errorf("FindAllNodesThatFitPod() (-want,+got):\n%s", diff)
	}
}

func TestCycleProvider(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	featuregatetesting.SetFeatureGateDuringTest(t, feature.DefaultFeatureGate, features.OpportunisticBatching, true)

	allNodes := []*v1.Node{
		st.MakeNode().Name("node1").Obj(),
	}
	snapshot := internalcache.NewSnapshot(nil, allNodes)
	algo := NewSchedulingAlgorithm(snapshot, nil, func() int64 { return 42 })

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

	_, _, err = algo.FindAllNodesThatFitPod(ctx, framework.NewCycleState(), recordingFwk, queuedPodInfo)
	if err != nil {
		t.Fatalf("FindAllNodesThatFitPod returned unexpected error: %v", err)
	}

	if recordingFwk.recordedCycle != 42 {
		t.Errorf("GetNodeHint received cycleCount = %d, want 42", recordingFwk.recordedCycle)
	}
}

type cycleRecordingFramework struct {
	framework.Framework
	recordedCycle int64
}

func (f *cycleRecordingFramework) GetNodeHint(ctx context.Context, pod *v1.Pod, signature fwk.PodSignature, state fwk.CycleState, cycleCount int64) string {
	f.recordedCycle = cycleCount
	return f.Framework.GetNodeHint(ctx, pod, signature, state, cycleCount)
}

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

// zeroCycle is the cycle provider for tests that do not exercise opportunistic
// batching, and so do not care which cycle the algorithm reports.
func zeroCycle() int64 { return 0 }

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

func nodeNamesInOrder(nodes []fwk.NodeInfo) []string {
	if len(nodes) == 0 {
		return nil
	}
	names := make([]string, 0, len(nodes))
	for _, n := range nodes {
		names = append(names, n.Node().Name)
	}
	return names
}

func nodeNamesFromNodeInfos(nodes []fwk.NodeInfo) []string {
	if len(nodes) == 0 {
		return nil
	}
	names := make([]string, 0, len(nodes))
	for _, n := range nodes {
		names = append(names, n.Node().Name)
	}
	sort.Strings(names)
	return names
}
