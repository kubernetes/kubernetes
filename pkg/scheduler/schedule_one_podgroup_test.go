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
	"fmt"
	"sort"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	v1 "k8s.io/api/core/v1"
	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/events"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	schedulingapi "k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	internalcache "k8s.io/kubernetes/pkg/scheduler/backend/cache"
	fakecache "k8s.io/kubernetes/pkg/scheduler/backend/cache/fake"
	internalqueue "k8s.io/kubernetes/pkg/scheduler/backend/queue"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultbinder"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/queuesort"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	"k8s.io/kubernetes/pkg/scheduler/profile"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	tf "k8s.io/kubernetes/pkg/scheduler/testing/framework"
)

// fakePodGroupPlugin simulates Filter, PostFilter, Permit and PodGroupPostFilter behaviors for PodGroup scheduling testing.
type fakePodGroupPlugin struct {
	filterStatus             map[string]*fwk.Status
	postFilterResult         map[string]*fwk.PostFilterResult
	postFilterStatus         map[string]*fwk.Status
	postFilterCalled         bool
	podGroupPostFilterStatus *fwk.Status
	podGroupPostFilterCalled bool
	podGroupPostFilterResult map[string]*fwk.NominatingInfo
}

var _ fwk.FilterPlugin = &fakePodGroupPlugin{}
var _ fwk.PostFilterPlugin = &fakePodGroupPlugin{}
var _ fwk.PodGroupPostFilterPlugin = &fakePodGroupPlugin{}

func (mp *fakePodGroupPlugin) Name() string { return "FakePodGroupPlugin" }

func (mp *fakePodGroupPlugin) Filter(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeInfo fwk.NodeInfo) *fwk.Status {
	if status, ok := mp.filterStatus[pod.Name]; ok {
		return status
	}
	return fwk.NewStatus(fwk.Unschedulable, "default fake filter failure")
}

func (mp *fakePodGroupPlugin) PostFilter(ctx context.Context, state fwk.CycleState, pod *v1.Pod, filteredNodeStatusMap fwk.NodeToStatusReader) (*fwk.PostFilterResult, *fwk.Status) {
	mp.postFilterCalled = true
	if status, ok := mp.postFilterStatus[pod.Name]; ok {
		return mp.postFilterResult[pod.Name], status
	}
	return &fwk.PostFilterResult{NominatingInfo: clearNominatedNode}, fwk.NewStatus(fwk.Unschedulable, "default fake postfilter failure")
}

func (mp *fakePodGroupPlugin) Permit(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) (*fwk.Status, time.Duration) {
	return fwk.NewStatus(fwk.Error, "unexpected call to permit"), 0
}

func (mp *fakePodGroupPlugin) PodGroupPostFilter(ctx context.Context, state fwk.PodGroupCycleState, pgInfo fwk.PodGroupInfo, pgSchedulingFunc fwk.PodGroupSchedulingFunc) (*fwk.PodGroupPostFilterResult, *fwk.Status) {
	mp.podGroupPostFilterCalled = true
	if mp.podGroupPostFilterStatus == nil {
		return nil, fwk.NewStatus(fwk.Unschedulable, "default fake podgroup postfilter failure")
	}
	if mp.podGroupPostFilterResult == nil {
		return nil, mp.podGroupPostFilterStatus
	}
	pods := pgInfo.GetUnscheduledPods()
	n := make(map[types.NamespacedName]*fwk.NominatingInfo, len(pods))
	for _, passedPod := range pods {
		namespacedName := types.NamespacedName{Namespace: passedPod.Namespace, Name: passedPod.Name}
		n[namespacedName] = mp.podGroupPostFilterResult[passedPod.Name]
	}
	return &fwk.PodGroupPostFilterResult{NominatingInfos: n}, mp.podGroupPostFilterStatus
}

type fakePlacementFeasiblePlugin struct {
	placementFeasibleStatuses [][]fwk.Code
	placementCount            int
}

var _ framework.PlacementFeasiblePlugin = &fakePlacementFeasiblePlugin{}
var _ fwk.PermitPlugin = &fakePlacementFeasiblePlugin{}

func (mp *fakePlacementFeasiblePlugin) Name() string {
	// Name has to be GangScheduling for the PlacementFeasible plugin to be used.
	// TODO: Remove this once the restriction is taken off.
	return names.GangScheduling
}

// PlacementFeasible simulates the evaluation of pod group placement constraints.
// The mock uses a 2D slice (placementFeasibleStatuses) where:
// - The outer slice represents distinct placements (e.g., when evaluating multiple topology placements).
// - The inner slice represents the pod-by-pod evaluation within a single placement.
func (mp *fakePlacementFeasiblePlugin) PlacementFeasible(ctx context.Context, placementCycleState fwk.PlacementCycleState, podGroupInfo fwk.PodGroupInfo, args framework.PlacementFeasibleArgs) *fwk.Status {
	// If no mock statuses are configured, always succeed.
	if len(mp.placementFeasibleStatuses) == 0 {
		return nil
	}

	if args.Evaluated == 0 {
		mp.placementCount++
	}

	placementIndex := mp.placementCount - 1

	// Ensure the indices are within the bounds of the injected statuses.
	if placementIndex < len(mp.placementFeasibleStatuses) {
		// If the specific placement has no pod statuses configured, treat it as always successful.
		if len(mp.placementFeasibleStatuses[placementIndex]) == 0 {
			return nil
		}
		if args.Evaluated < len(mp.placementFeasibleStatuses[placementIndex]) {
			code := mp.placementFeasibleStatuses[placementIndex][args.Evaluated]
			if code == fwk.Success {
				return nil
			}
			return fwk.NewStatus(code, "injected placementFeasible status")
		}
	}
	return fwk.AsStatus(fmt.Errorf("exceeded the expected call count"))
}

func (mp *fakePlacementFeasiblePlugin) Permit(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) (*fwk.Status, time.Duration) {
	return fwk.NewStatus(fwk.Error, "unexpected call to permit"), 0
}

func TestValidatePodGroup(t *testing.T) {
	tests := []struct {
		name          string
		podGroup      *schedulingv1alpha3.PodGroup
		scheduledPods []*v1.Pod
		pods          []*v1.Pod
		profiles      profile.Map
		expectError   bool
	}{
		{
			name:     "failure when no pods to evaluate",
			podGroup: st.MakePodGroup().Name("pg").Obj(),
			pods:     []*v1.Pod{},
			profiles: profile.Map{
				"sched1": nil,
			},
			expectError: true,
		},
		{
			name:     "success for same scheduler name",
			podGroup: st.MakePodGroup().Name("pg").Obj(),
			pods: []*v1.Pod{
				st.MakePod().Name("p1").PodGroupName("pg").SchedulerName("sched1").Obj(),
				st.MakePod().Name("p2").PodGroupName("pg").SchedulerName("sched1").Obj(),
			},
			profiles: profile.Map{
				"sched1": nil,
			},
			expectError: false,
		},
		{
			name:     "failure for different scheduler names",
			podGroup: st.MakePodGroup().Name("pg").Obj(),
			pods: []*v1.Pod{
				st.MakePod().Name("p1").PodGroupName("pg").SchedulerName("sched1").Obj(),
				st.MakePod().Name("p2").PodGroupName("pg").SchedulerName("sched2").Obj(),
			},
			profiles: profile.Map{
				"sched1": nil,
				"sched2": nil,
			},
			expectError: true,
		},
		{
			name:     "failure when profile not found",
			podGroup: st.MakePodGroup().Name("pg").Obj(),
			pods: []*v1.Pod{
				st.MakePod().Name("p1").PodGroupName("pg").SchedulerName("sched1").Obj(),
				st.MakePod().Name("p2").PodGroupName("pg").SchedulerName("sched1").Obj(),
			},
			profiles: profile.Map{
				"other": nil,
			},
			expectError: true,
		},
		{
			name:     "success when priorities match",
			podGroup: st.MakePodGroup().Name("pg").Priority(10).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Name("p1").PodGroupName("pg").Priority(10).Obj(),
				st.MakePod().Name("p2").PodGroupName("pg").Priority(10).Obj(),
			},
			profiles: profile.Map{
				"": nil,
			},
			expectError: false,
		},
		{
			name:     "failure when different priorities across pods",
			podGroup: st.MakePodGroup().Name("pg").Priority(10).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Name("p1").PodGroupName("pg").Priority(9).Obj(),
				st.MakePod().Name("p2").PodGroupName("pg").Priority(10).Obj(),
			},
			profiles: profile.Map{
				"": nil,
			},
			expectError: true,
		},
		{
			name:     "failure when different priorities across pods and pod group",
			podGroup: st.MakePodGroup().Name("pg").Priority(9).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Name("p1").PodGroupName("pg").Priority(10).Obj(),
				st.MakePod().Name("p2").PodGroupName("pg").Priority(10).Obj(),
			},
			profiles: profile.Map{
				"": nil,
			},
			expectError: true,
		},
		{
			name:     "success when new pods match scheduled pods scheduler name and priority",
			podGroup: st.MakePodGroup().Name("pg").Priority(10).Obj(),
			scheduledPods: []*v1.Pod{
				st.MakePod().Name("p2").UID("p2").PodGroupName("pg").Priority(10).SchedulerName("sched1").Node("node1").Obj(),
			},
			pods: []*v1.Pod{
				st.MakePod().Name("p1").UID("p1").PodGroupName("pg").Priority(10).SchedulerName("sched1").Obj(),
			},
			profiles: profile.Map{
				"sched1": nil,
			},
			expectError: false,
		},
		{
			name:     "failure when new pod has different scheduler name than scheduled pod",
			podGroup: st.MakePodGroup().Name("pg").Priority(10).Obj(),
			scheduledPods: []*v1.Pod{
				st.MakePod().Name("p2").UID("p2").PodGroupName("pg").Priority(10).SchedulerName("sched2").Node("node1").Obj(),
			},
			pods: []*v1.Pod{
				st.MakePod().Name("p1").UID("p1").PodGroupName("pg").Priority(10).SchedulerName("sched1").Obj(),
			},
			profiles: profile.Map{
				"sched1": nil,
			},
			expectError: true,
		},
		{
			name:     "failure when new pod has different priority than scheduled pod",
			podGroup: st.MakePodGroup().Name("pg").Priority(10).Obj(),
			scheduledPods: []*v1.Pod{
				st.MakePod().Name("p2").UID("p2").PodGroupName("pg").Priority(9).SchedulerName("sched1").Node("node1").Obj(),
			},
			pods: []*v1.Pod{
				st.MakePod().Name("p1").UID("p1").PodGroupName("pg").Priority(10).SchedulerName("sched1").Obj(),
			},
			profiles: profile.Map{
				"sched1": nil,
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload: true,
			})
			snapshot := internalcache.NewTestSnapshotWithPodGroups(tt.scheduledPods, nil, []*schedulingv1alpha3.PodGroup{tt.podGroup})
			sched := &Scheduler{
				Profiles:         tt.profiles,
				nodeInfoSnapshot: snapshot,
			}

			podGroupInfo := &framework.QueuedPodGroupInfo{
				PodGroupInfo: &framework.PodGroupInfo{
					Name:      tt.podGroup.Name,
					Namespace: tt.podGroup.Namespace,
					PodGroup:  tt.podGroup,
				},
			}
			for _, pod := range tt.pods {
				podGroupInfo.UnscheduledPods = append(podGroupInfo.UnscheduledPods, pod)
				podGroupInfo.QueuedPodInfos = append(podGroupInfo.QueuedPodInfos,
					&framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: pod}})
			}
			err := sched.validatePodGroup(podGroupInfo)
			if tt.expectError {
				if err == nil {
					t.Errorf("Expected error, but got nil")
				}
			} else {
				if err != nil {
					t.Errorf("Expected no error, but got: %v", err)
				}
			}
		})
	}
}

func TestSkipPodGroupPodSchedule(t *testing.T) {
	p1 := st.MakePod().Name("p1").UID("p1").PodGroupName("pg").Obj()
	p2 := st.MakePod().Name("p2").UID("p2").PodGroupName("pg").Terminating().Obj()
	p3 := st.MakePod().Name("p3").UID("p3").PodGroupName("pg").Obj()

	qInfo1 := &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: p1}}
	qInfo2 := &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: p2}}
	qInfo3 := &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: p3}}

	podGroupInfo := &framework.QueuedPodGroupInfo{
		QueuedPodInfos: []*framework.QueuedPodInfo{qInfo1, qInfo2, qInfo3},
		PodGroupInfo: &framework.PodGroupInfo{
			UnscheduledPods: []*v1.Pod{p1, p2, p3},
		},
	}

	logger, ctx := ktesting.NewTestContext(t)

	cache := internalcache.New(ctx, nil, true)
	registry := frameworkruntime.Registry{
		queuesort.Name:     queuesort.New,
		defaultbinder.Name: defaultbinder.New,
	}
	profileCfg := config.KubeSchedulerProfile{
		SchedulerName: "default-scheduler",
		Plugins: &config.Plugins{
			QueueSort: config.PluginSet{
				Enabled: []config.Plugin{{Name: queuesort.Name}},
			},
			Bind: config.PluginSet{
				Enabled: []config.Plugin{{Name: defaultbinder.Name}},
			},
		},
	}
	schedFwk, err := frameworkruntime.NewFramework(ctx, registry, &profileCfg,
		frameworkruntime.WithEventRecorder(events.NewFakeRecorder(100)),
	)
	if err != nil {
		t.Fatalf("Failed to create new framework: %v", err)
	}
	sched := &Scheduler{
		SchedulingQueue: internalqueue.NewTestQueue(ctx, nil),
		Cache:           cache,
	}

	// Assume pod p3
	err = cache.AssumePod(logger, p3)
	if err != nil {
		t.Fatalf("Failed to assume pod p3: %v", err)
	}

	sched.skipPodGroupPodSchedule(ctx, schedFwk, podGroupInfo)

	if len(podGroupInfo.QueuedPodInfos) != 1 {
		t.Errorf("Expected 1 queued pod left, got %d", len(podGroupInfo.QueuedPodInfos))
	}
	if podGroupInfo.QueuedPodInfos[0].Pod.Name != "p1" {
		t.Errorf("Expected p1 to be left in queued pods, got %s", podGroupInfo.QueuedPodInfos[0].Pod.Name)
	}
	if len(podGroupInfo.UnscheduledPods) != 1 {
		t.Errorf("Expected 1 unscheduled pod left, got %d", len(podGroupInfo.UnscheduledPods))
	}
	if podGroupInfo.UnscheduledPods[0].Name != "p1" {
		t.Errorf("Expected p1 to be left in unscheduled pods, got %s", podGroupInfo.UnscheduledPods[0].Name)
	}
}

func TestPodGroupCycle_UpdateSnapshotError(t *testing.T) {
	p1 := st.MakePod().Name("p1").UID("p1").PodGroupName("pg").SchedulerName("test-scheduler").Obj()
	p2 := st.MakePod().Name("p2").UID("p2").PodGroupName("pg").SchedulerName("test-scheduler").Obj()
	qInfo1 := &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: p1}}
	qInfo2 := &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: p2}}

	testPodGroup := &schedulingv1alpha3.PodGroup{
		ObjectMeta: metav1.ObjectMeta{Name: "pg", Namespace: "default"},
	}

	podGroupInfo := &framework.QueuedPodGroupInfo{
		QueuedPodInfos: []*framework.QueuedPodInfo{qInfo1, qInfo2},
		PodGroupInfo: &framework.PodGroupInfo{
			Name:            "pg",
			Namespace:       "default",
			UnscheduledPods: []*v1.Pod{p1, p2},
			PodGroup:        testPodGroup,
		},
	}

	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	registry := frameworkruntime.Registry{
		queuesort.Name:     queuesort.New,
		defaultbinder.Name: defaultbinder.New,
	}
	profileCfg := config.KubeSchedulerProfile{
		SchedulerName: "test-scheduler",
		Plugins: &config.Plugins{
			QueueSort: config.PluginSet{
				Enabled: []config.Plugin{{Name: queuesort.Name}},
			},
			Bind: config.PluginSet{
				Enabled: []config.Plugin{{Name: defaultbinder.Name}},
			},
		},
	}
	schedFwk, err := frameworkruntime.NewFramework(ctx, registry, &profileCfg,
		frameworkruntime.WithEventRecorder(events.NewFakeRecorder(100)),
	)
	if err != nil {
		t.Fatalf("Failed to create new framework: %v", err)
	}

	// Create fake cache that returns error on UpdateSnapshot
	updateSnapshotErr := fmt.Errorf("update snapshot error")
	cache := &fakecache.Cache{
		Cache: internalcache.New(ctx, nil, true),
		UpdateSnapshotFunc: func(nodeSnapshot *internalcache.Snapshot) error {
			return updateSnapshotErr
		},
	}

	client := clientsetfake.NewClientset(testPodGroup)
	informerFactory := informers.NewSharedInformerFactory(client, 0)
	informerFactory.Start(ctx.Done())
	informerFactory.WaitForCacheSync(ctx.Done())

	var failureHandlerCalled bool
	sched := &Scheduler{
		Profiles:        profile.Map{"test-scheduler": schedFwk},
		SchedulingQueue: internalqueue.NewTestQueue(ctx, nil),
		Cache:           cache,
		client:          client,
		FailureHandler: func(ctx context.Context, fwk framework.Framework, p *framework.QueuedPodInfo, status *fwk.Status, ni *fwk.NominatingInfo, start time.Time) {
			failureHandlerCalled = true
			if updateSnapshotErr.Error() != status.AsError().Error() {
				t.Errorf("Expected status error %q, got %q", updateSnapshotErr, status.AsError())
			}
		},
	}

	sched.scheduleOnePodGroup(ctx, podGroupInfo)

	if !failureHandlerCalled {
		t.Errorf("Expected FailureHandler to be called after UpdateSnapshot failed")
	}
}

func TestPodGroupCycle_FillsPodResultsOnFewerResults(t *testing.T) {
	testPodGroup := st.MakePodGroup().Name("pg").Namespace("default").Obj()
	p1 := st.MakePod().Name("p1").UID("p1").PodGroupName("pg").SchedulerName("test-scheduler").Obj()
	p2 := st.MakePod().Name("p2").UID("p2").PodGroupName("pg").SchedulerName("test-scheduler").Obj()
	p3 := st.MakePod().Name("p3").UID("p3").PodGroupName("pg").SchedulerName("test-scheduler").Obj()
	testNode := st.MakeNode().Name("node1").UID("node1").Obj()

	qInfo1 := &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: p1}}
	qInfo2 := &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: p2}}
	qInfo3 := &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: p3}}

	podGroupInfo := &framework.QueuedPodGroupInfo{
		QueuedPodInfos: []*framework.QueuedPodInfo{qInfo1, qInfo2, qInfo3},
		PodGroupInfo: &framework.PodGroupInfo{
			Name:            "pg",
			Namespace:       "default",
			UnscheduledPods: []*v1.Pod{p1, p2, p3},
			PodGroup:        testPodGroup,
		},
	}

	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	fakePlugin := &fakePodGroupPlugin{
		filterStatus: map[string]*fwk.Status{
			"p1": nil,
			"p2": fwk.NewStatus(fwk.Error, "filter error for p2"),
			"p3": nil,
		},
	}

	registry := []tf.RegisterPluginFunc{
		tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
		tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
		tf.RegisterPostFilterPlugin(fakePlugin.Name(), func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
			return fakePlugin, nil
		}),
		tf.RegisterPermitPlugin(fakePlugin.Name(), func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
			return fakePlugin, nil
		}),
		tf.RegisterFilterPlugin(fakePlugin.Name(), func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
			return fakePlugin, nil
		}),
	}

	client := clientsetfake.NewSimpleClientset(testPodGroup, testNode)
	informerFactory := informers.NewSharedInformerFactory(client, 0)

	informerFactory.Start(ctx.Done())
	informerFactory.WaitForCacheSync(ctx.Done())
	queue := internalqueue.NewSchedulingQueue(nil, informerFactory)
	snapshot := internalcache.NewEmptySnapshot()

	schedFwk, err := tf.NewFramework(ctx, registry, "test-scheduler",
		frameworkruntime.WithInformerFactory(informerFactory),
		frameworkruntime.WithSnapshotSharedLister(snapshot),
		frameworkruntime.WithPodNominator(queue),
		frameworkruntime.WithClientSet(client),
		frameworkruntime.WithEventRecorder(events.NewFakeRecorder(100)),
	)
	if err != nil {
		t.Fatalf("Failed to create new framework: %v", err)
	}

	cache := internalcache.New(ctx, nil, true)
	logger, ctx := ktesting.NewTestContext(t)
	cache.AddNode(logger, testNode)

	handledPods := make(map[string]*fwk.Status)
	var lock sync.Mutex

	sched := &Scheduler{
		Profiles:         profile.Map{"test-scheduler": schedFwk},
		SchedulingQueue:  internalqueue.NewTestQueue(ctx, nil),
		Cache:            cache,
		client:           client,
		nodeInfoSnapshot: internalcache.NewEmptySnapshot(),
		FailureHandler: func(ctx context.Context, fwk framework.Framework, p *framework.QueuedPodInfo, status *fwk.Status, ni *fwk.NominatingInfo, start time.Time) {
			lock.Lock()
			defer lock.Unlock()
			handledPods[p.Pod.Name] = status
		},
	}

	// Checking that scheduling algorithm is returning shorter list
	if err := sched.Cache.UpdateSnapshot(logger, sched.nodeInfoSnapshot); err != nil {
		t.Fatalf("Failed to update snapshot: %v", err)
	}
	sched.SchedulePod = sched.schedulePod
	schedulePodResult := sched.podGroupSchedulingAlgorithm(ctx, schedFwk, framework.NewCycleState(), podGroupInfo, runAllPostFilters)
	if len(schedulePodResult.podResults) != 2 {
		t.Errorf("Expected 2 pod results, got %d", len(schedulePodResult.podResults))
	}

	// Run the scheduling cycle and check that all pods are handled.
	sched.podGroupCycle(ctx, schedFwk, framework.NewCycleState(), podGroupInfo, time.Now())

	lock.Lock()
	defer lock.Unlock()

	if len(handledPods) != 3 {
		t.Errorf("Expected FailureHandler to be called for 3 pods, but got called for %d", len(handledPods))
	}

	expectedGroupErrMsg := "failed to schedule other pod from a pod group: running \"FakePodGroupPlugin\" filter plugin: filter error for p2"
	expectedP2ErrMsg := "running \"FakePodGroupPlugin\" filter plugin: filter error for p2"

	if status, ok := handledPods["p1"]; !ok {
		t.Errorf("Expected FailureHandler to be called for p1")
	} else if status.AsError() == nil || status.AsError().Error() != expectedGroupErrMsg {
		t.Errorf("Expected status error for p1 to be %q, got %v", expectedGroupErrMsg, status.AsError())
	}

	if status, ok := handledPods["p2"]; !ok {
		t.Errorf("Expected FailureHandler to be called for p2")
	} else if status.AsError() == nil || status.AsError().Error() != expectedP2ErrMsg {
		t.Errorf("Expected status error for p2 to be %q, got %v", expectedP2ErrMsg, status.AsError())
	}

	if status, ok := handledPods["p3"]; !ok {
		t.Errorf("Expected FailureHandler to be called for p3")
	} else if status.AsError() == nil || status.AsError().Error() != expectedGroupErrMsg {
		t.Errorf("Expected status error for p3 to be %q, got %v", expectedGroupErrMsg, status.AsError())
	}
}

func TestPodGroupCycle_PodGroupPostFilter(t *testing.T) {
	tests := []struct {
		name                             string
		genericWorkloadEnabled           bool
		postFilterPlugin                 string
		expectedPodGroupPostFilterCalled bool
	}{
		{
			name:                             "runs pod group post filter when GenericWorkload is enabled and DefaultPreemption is registered",
			genericWorkloadEnabled:           true,
			postFilterPlugin:                 "DefaultPreemption",
			expectedPodGroupPostFilterCalled: true,
		},
		{
			name:                             "disables pod group post filter when GenericWorkload feature gate is disabled",
			genericWorkloadEnabled:           false,
			postFilterPlugin:                 "DefaultPreemption",
			expectedPodGroupPostFilterCalled: false,
		},
		{
			name:                             "disables pod group post filter when DefaultPreemption is not registered",
			genericWorkloadEnabled:           true,
			postFilterPlugin:                 "FakePodGroupPlugin",
			expectedPodGroupPostFilterCalled: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload: tt.genericWorkloadEnabled,
			})

			testPodGroup := st.MakePodGroup().Name("pg").Namespace("default").Obj()
			p1 := st.MakePod().Name("p1").UID("p1").PodGroupName("pg").SchedulerName("test-scheduler").Obj()
			p2 := st.MakePod().Name("p2").UID("p2").PodGroupName("pg").SchedulerName("test-scheduler").Obj()
			testNode := st.MakeNode().Name("node1").UID("node1").Obj()

			qInfo1 := &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: p1}}
			qInfo2 := &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: p2}}

			podGroupInfo := &framework.QueuedPodGroupInfo{
				QueuedPodInfos: []*framework.QueuedPodInfo{qInfo1, qInfo2},
				PodGroupInfo: &framework.PodGroupInfo{
					Name:            "pg",
					Namespace:       "default",
					UnscheduledPods: []*v1.Pod{p1, p2},
					PodGroup:        testPodGroup,
				},
			}

			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			fakePlugin := &fakePodGroupPlugin{
				filterStatus: map[string]*fwk.Status{
					"p1": fwk.NewStatus(fwk.Unschedulable, "always fail p1"),
					"p2": fwk.NewStatus(fwk.Unschedulable, "always fail p2"),
				},
				podGroupPostFilterStatus: nil,
			}

			registry := frameworkruntime.Registry{
				queuesort.Name:     queuesort.New,
				defaultbinder.Name: defaultbinder.New,
				"FakePodGroupPlugin": func(ctx context.Context, obj runtime.Object, handle fwk.Handle) (fwk.Plugin, error) {
					return fakePlugin, nil
				},
			}

			if tt.postFilterPlugin == "DefaultPreemption" {
				// Register the same plugin as DefaultPreemption to fulfill runWorkloadAwarePreemption requirements.
				registry["DefaultPreemption"] = func(ctx context.Context, obj runtime.Object, handle fwk.Handle) (fwk.Plugin, error) {
					return &fakeDefaultPreemption{fakePodGroupPlugin: fakePlugin}, nil
				}
			}

			var pgPostFilterPlugins []config.Plugin
			if tt.postFilterPlugin == "DefaultPreemption" {
				pgPostFilterPlugins = []config.Plugin{{Name: "DefaultPreemption"}}
			}

			profileCfg := config.KubeSchedulerProfile{
				SchedulerName: "test-scheduler",
				Plugins: &config.Plugins{
					QueueSort: config.PluginSet{
						Enabled: []config.Plugin{{Name: queuesort.Name}},
					},
					Filter: config.PluginSet{
						Enabled: []config.Plugin{{Name: "FakePodGroupPlugin"}},
					},
					Bind: config.PluginSet{
						Enabled: []config.Plugin{{Name: defaultbinder.Name}},
					},
					PostFilter: config.PluginSet{
						Enabled: []config.Plugin{{Name: tt.postFilterPlugin}},
					},
					PodGroupPostFilter: config.PluginSet{
						Enabled: pgPostFilterPlugins,
					},
				},
			}

			client := clientsetfake.NewSimpleClientset(testPodGroup, testNode)
			informerFactory := informers.NewSharedInformerFactory(client, 0)

			informerFactory.Start(ctx.Done())
			informerFactory.WaitForCacheSync(ctx.Done())
			queue := internalqueue.NewSchedulingQueue(nil, informerFactory)
			snapshot := internalcache.NewEmptySnapshot()

			schedFwk, err := frameworkruntime.NewFramework(ctx, registry, &profileCfg,
				frameworkruntime.WithInformerFactory(informerFactory),
				frameworkruntime.WithSnapshotSharedLister(snapshot),
				frameworkruntime.WithPodNominator(queue),
				frameworkruntime.WithClientSet(client),
				frameworkruntime.WithEventRecorder(events.NewFakeRecorder(100)),
			)
			if err != nil {
				t.Fatalf("Failed to create new framework: %v", err)
			}

			cache := internalcache.New(ctx, nil, true)
			logger, ctx := ktesting.NewTestContext(t)
			cache.AddNode(logger, testNode)

			sched := &Scheduler{
				Profiles:               profile.Map{"test-scheduler": schedFwk},
				SchedulingQueue:        internalqueue.NewTestQueue(ctx, nil),
				Cache:                  cache,
				client:                 client,
				nodeInfoSnapshot:       internalcache.NewEmptySnapshot(),
				genericWorkloadEnabled: tt.genericWorkloadEnabled,
				FailureHandler: func(ctx context.Context, fwk framework.Framework, p *framework.QueuedPodInfo, status *fwk.Status, ni *fwk.NominatingInfo, start time.Time) {
				},
			}

			sched.SchedulePod = sched.schedulePod
			sched.podGroupCycle(ctx, schedFwk, framework.NewCycleState(), podGroupInfo, time.Now())

			if fakePlugin.podGroupPostFilterCalled != tt.expectedPodGroupPostFilterCalled {
				t.Errorf("Expected workload aware preemption (PodGroupPostFilter) to be %v, but got %v", tt.expectedPodGroupPostFilterCalled, fakePlugin.podGroupPostFilterCalled)
			}
		})
	}
}

func TestPodGroupSchedulingAlgorithm(t *testing.T) {
	testNode := st.MakeNode().Name("node1").UID("node1").Obj()

	p1 := st.MakePod().Name("p1").UID("p1").PodGroupName("pg").SchedulerName("test-scheduler").Obj()
	p2 := st.MakePod().Name("p2").UID("p2").PodGroupName("pg").SchedulerName("test-scheduler").Obj()
	p3 := st.MakePod().Name("p3").UID("p3").PodGroupName("pg").SchedulerName("test-scheduler").Obj()

	qInfo1 := &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: p1}}
	qInfo2 := &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: p2}}
	qInfo3 := &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: p3}}

	podGroupInfo := &framework.QueuedPodGroupInfo{
		QueuedPodInfos: []*framework.QueuedPodInfo{qInfo1, qInfo2, qInfo3},
		PodGroupInfo: &framework.PodGroupInfo{
			Name:            "pg",
			UnscheduledPods: []*v1.Pod{p1, p2, p3},
		},
	}

	tests := []struct {
		name                             string
		plugin                           *fakePodGroupPlugin
		podGroupFeasibleStatuses         []fwk.Code
		expectedGroupStatusCode          fwk.Code
		expectedGroupWaitingOnPreemption bool
		expectedPodStatus                map[string]*fwk.Status
		expectedPreemption               map[string]bool
		postFilterMode                   podGroupPostFilterMode
		skipForTAS                       bool
	}{
		{
			name: "All pods feasible",
			plugin: &fakePodGroupPlugin{
				filterStatus: map[string]*fwk.Status{
					"p1": nil,
					"p2": nil,
					"p3": nil,
				},
			},
			expectedGroupStatusCode: fwk.Success,
			expectedPodStatus: map[string]*fwk.Status{
				"p1": nil,
				"p2": nil,
				"p3": nil,
			},
		},
		{
			name: "All pods feasible, podGroup already meeting quorum before any pod is evaluated",
			plugin: &fakePodGroupPlugin{
				filterStatus: map[string]*fwk.Status{
					"p1": nil,
					"p2": nil,
					"p3": nil,
				},
			},
			podGroupFeasibleStatuses: []fwk.Code{
				fwk.Success,
				fwk.Success,
				fwk.Success,
				fwk.Success,
			},
			expectedGroupStatusCode: fwk.Success,
			expectedPodStatus: map[string]*fwk.Status{
				"p1": nil,
				"p2": nil,
				"p3": nil,
			},
		},
		{
			name: "All pods feasible, podGroup schedulable with 3 schedulable pods",
			plugin: &fakePodGroupPlugin{
				filterStatus: map[string]*fwk.Status{
					"p1": nil,
					"p2": nil,
					"p3": nil,
				},
			},
			podGroupFeasibleStatuses: []fwk.Code{
				fwk.Wait,
				fwk.Wait,
				fwk.Wait,
				fwk.Success,
			},
			expectedGroupStatusCode: fwk.Success,
			expectedPodStatus: map[string]*fwk.Status{
				"p1": nil,
				"p2": nil,
				"p3": nil,
			},
		},
		{
			name: "All pods feasible, podGroup waiting",
			plugin: &fakePodGroupPlugin{
				filterStatus: map[string]*fwk.Status{
					"p1": nil,
					"p2": nil,
					"p3": nil,
				},
			},
			podGroupFeasibleStatuses: []fwk.Code{
				fwk.Wait,
				fwk.Wait,
				fwk.Wait,
				fwk.Wait,
			},
			expectedGroupStatusCode: fwk.Unschedulable,
			expectedPodStatus: map[string]*fwk.Status{
				"p1": nil,
				"p2": nil,
				"p3": nil,
			},
		},
		{
			name: "All pods feasible, podGroup unschedulable",
			plugin: &fakePodGroupPlugin{
				filterStatus: map[string]*fwk.Status{
					"p1": nil,
					"p2": nil,
					"p3": nil,
				},
			},
			podGroupFeasibleStatuses: []fwk.Code{
				fwk.Wait,
				fwk.Unschedulable,
			},
			expectedGroupStatusCode: fwk.Unschedulable,
			expectedPodStatus: map[string]*fwk.Status{
				"p1": nil,
				// The algorithm stopped evaluating the pods after Unschedulable was received from PlacementFeasible.
			},
		},
		{
			name: "All pods feasible, podGroup unschedulable with 2 pods",
			plugin: &fakePodGroupPlugin{
				filterStatus: map[string]*fwk.Status{
					"p1": nil,
					"p2": nil,
					"p3": nil,
				},
			},
			podGroupFeasibleStatuses: []fwk.Code{
				fwk.Wait,
				fwk.Wait,
				fwk.Unschedulable,
			},
			expectedGroupStatusCode: fwk.Unschedulable,
			expectedPodStatus: map[string]*fwk.Status{
				"p1": nil,
				"p2": nil,
				// The algorithm stopped evaluating the pods after Unschedulable was received from PlacementFeasible.
			},
		},
		{
			name: "All pods require preemption",
			plugin: &fakePodGroupPlugin{
				filterStatus: map[string]*fwk.Status{
					"p1": fwk.NewStatus(fwk.Unschedulable),
					"p2": fwk.NewStatus(fwk.Unschedulable),
					"p3": fwk.NewStatus(fwk.Unschedulable),
				},
				postFilterStatus: map[string]*fwk.Status{
					"p1": nil,
					"p2": nil,
					"p3": nil,
				},
				postFilterResult: map[string]*fwk.PostFilterResult{
					"p1": {NominatingInfo: &fwk.NominatingInfo{NominatedNodeName: "node1", NominatingMode: fwk.ModeOverride}},
					"p2": {NominatingInfo: &fwk.NominatingInfo{NominatedNodeName: "node1", NominatingMode: fwk.ModeOverride}},
					"p3": {NominatingInfo: &fwk.NominatingInfo{NominatedNodeName: "node1", NominatingMode: fwk.ModeOverride}},
				},
			},
			expectedGroupStatusCode:          fwk.Unschedulable,
			expectedGroupWaitingOnPreemption: true,
			expectedPodStatus: map[string]*fwk.Status{
				"p1": fwk.NewStatus(fwk.Unschedulable),
				"p2": fwk.NewStatus(fwk.Unschedulable),
				"p3": fwk.NewStatus(fwk.Unschedulable),
			},
			expectedPreemption: map[string]bool{
				"p1": true,
				"p2": true,
				"p3": true,
			},
			skipForTAS: true,
		},
		{
			name: "One pod requires preemption, podGroup schedulable with 2 schedulable pods",
			plugin: &fakePodGroupPlugin{
				filterStatus: map[string]*fwk.Status{
					"p1": fwk.NewStatus(fwk.Unschedulable),
					"p2": nil,
					"p3": nil,
				},
				postFilterStatus: map[string]*fwk.Status{
					"p1": nil,
				},
				postFilterResult: map[string]*fwk.PostFilterResult{
					"p1": {NominatingInfo: &fwk.NominatingInfo{NominatedNodeName: "node1", NominatingMode: fwk.ModeOverride}},
				},
			},
			podGroupFeasibleStatuses: []fwk.Code{
				fwk.Wait,
				fwk.Wait,
				fwk.Wait,
				fwk.Success,
			},
			expectedGroupStatusCode:          fwk.Unschedulable,
			expectedGroupWaitingOnPreemption: true,
			expectedPodStatus: map[string]*fwk.Status{
				"p1": fwk.NewStatus(fwk.Unschedulable),
				"p2": nil,
				"p3": nil,
			},
			expectedPreemption: map[string]bool{
				"p1": true,
				"p2": false,
				"p3": false,
			},
			// preemption is not yet implemented for TAS
			skipForTAS: true,
		},
		{
			name: "One pod unschedulable, one requires preemption, one feasible",
			plugin: &fakePodGroupPlugin{
				filterStatus: map[string]*fwk.Status{
					"p1": fwk.NewStatus(fwk.Unschedulable),
					"p2": fwk.NewStatus(fwk.Unschedulable),
					"p3": nil,
				},
				postFilterStatus: map[string]*fwk.Status{
					"p1": nil,
					"p2": fwk.NewStatus(fwk.Unschedulable),
				},
				postFilterResult: map[string]*fwk.PostFilterResult{
					"p1": {NominatingInfo: &fwk.NominatingInfo{NominatedNodeName: "node1", NominatingMode: fwk.ModeOverride}},
					"p2": {NominatingInfo: clearNominatedNode},
				},
			},
			expectedGroupStatusCode:          fwk.Unschedulable,
			expectedGroupWaitingOnPreemption: true,
			expectedPodStatus: map[string]*fwk.Status{
				"p1": fwk.NewStatus(fwk.Unschedulable),
				"p2": fwk.NewStatus(fwk.Unschedulable),
				"p3": nil,
			},
			expectedPreemption: map[string]bool{
				"p1": true,
				"p2": false,
				"p3": false,
			},
			// preemption is not yet implemented for TAS
			skipForTAS: true,
		},
		{
			name: "All pods unschedulable",
			plugin: &fakePodGroupPlugin{
				filterStatus: map[string]*fwk.Status{
					"p1": fwk.NewStatus(fwk.UnschedulableAndUnresolvable),
					"p2": fwk.NewStatus(fwk.Unschedulable),
					"p3": fwk.NewStatus(fwk.Unschedulable),
				},
			},
			podGroupFeasibleStatuses: []fwk.Code{
				fwk.Wait,
				fwk.Success,
				fwk.Success,
				fwk.Success,
			},
			expectedGroupStatusCode: fwk.Unschedulable,
			expectedPodStatus: map[string]*fwk.Status{
				"p1": fwk.NewStatus(fwk.Unschedulable),
				"p2": fwk.NewStatus(fwk.Unschedulable),
				"p3": fwk.NewStatus(fwk.Unschedulable),
			},
		},
		{
			name: "Any filter returned Error",
			plugin: &fakePodGroupPlugin{
				filterStatus: map[string]*fwk.Status{
					"p1": nil,
					"p2": fwk.NewStatus(fwk.Error),
					"p3": nil,
				},
			},
			expectedGroupStatusCode: fwk.Error,
			expectedPodStatus: map[string]*fwk.Status{
				"p1": nil,
				"p2": fwk.NewStatus(fwk.Error),
				// The algorithm stopped evaluating the pods after an error occurred, so a "p3" status is not expected.
			},
		},
		{
			name: "Any placementFeasible returned Error",
			plugin: &fakePodGroupPlugin{
				filterStatus: map[string]*fwk.Status{
					"p1": nil,
					"p2": nil,
					"p3": nil,
				},
			},
			podGroupFeasibleStatuses: []fwk.Code{
				fwk.Wait,
				fwk.Success,
				fwk.Error,
			},
			expectedGroupStatusCode: fwk.Error,
			expectedPodStatus: map[string]*fwk.Status{
				"p1": nil,
				"p2": nil,
				// The algorithm stopped evaluating the pods after an error occurred, so a "p3" status is not expected.
			},
		},
		{
			name: "First placementFeasible call returned Unschedulable",
			plugin: &fakePodGroupPlugin{
				filterStatus: map[string]*fwk.Status{
					"p1": nil,
					"p2": nil,
					"p3": nil,
				},
			},
			podGroupFeasibleStatuses: []fwk.Code{
				fwk.Unschedulable,
			},
			expectedGroupStatusCode: fwk.Unschedulable,
			expectedPodStatus:       map[string]*fwk.Status{
				// The algorithm didn't evaluate any pods whatsoever because the first call to PlacementFeasible Plugin returned Unschedulable.
			},
		},
		{
			name: "First placementFeasible call returned Error",
			plugin: &fakePodGroupPlugin{
				filterStatus: map[string]*fwk.Status{
					"p1": nil,
					"p2": nil,
					"p3": nil,
				},
			},
			podGroupFeasibleStatuses: []fwk.Code{
				fwk.Error,
			},
			expectedGroupStatusCode: fwk.Error,
			expectedPodStatus:       map[string]*fwk.Status{
				// The algorithm didn't evaluate any pods whatsoever because the first call to PlacementFeasible Plugin returned Error.
			},
		},
		{
			name: "Any filter returned Error while waiting on preemption",
			plugin: &fakePodGroupPlugin{
				filterStatus: map[string]*fwk.Status{
					"p1": fwk.NewStatus(fwk.Unschedulable),
					"p2": fwk.NewStatus(fwk.Error),
					"p3": nil,
				},
				postFilterResult: map[string]*fwk.PostFilterResult{
					"p1": {NominatingInfo: &fwk.NominatingInfo{NominatedNodeName: "node1", NominatingMode: fwk.ModeOverride}},
				},
			},
			expectedGroupStatusCode: fwk.Error,
			expectedPodStatus: map[string]*fwk.Status{
				"p1": fwk.NewStatus(fwk.Unschedulable),
				"p2": fwk.NewStatus(fwk.Error),
				// The algorithm stopped evaluating the pods after an error occurred, so a "p3" status is not expected.
			},
		},
		{
			name: "Any placementFeasible returned Error while waiting on preemption",
			plugin: &fakePodGroupPlugin{
				filterStatus: map[string]*fwk.Status{
					"p1": fwk.NewStatus(fwk.Unschedulable),
					"p2": nil,
					"p3": nil,
				},
				postFilterResult: map[string]*fwk.PostFilterResult{
					"p1": {NominatingInfo: &fwk.NominatingInfo{NominatedNodeName: "node1", NominatingMode: fwk.ModeOverride}},
				},
			},
			podGroupFeasibleStatuses: []fwk.Code{
				fwk.Wait,
				fwk.Success,
				fwk.Error,
			},
			expectedGroupStatusCode: fwk.Error,
			expectedPodStatus: map[string]*fwk.Status{
				"p1": fwk.NewStatus(fwk.Unschedulable),
				"p2": nil,
				// The algorithm stopped evaluating the pods after an error occurred, so a "p3" status is not expected.
			},
		},
	}

	for _, tasEnabled := range []bool{true, false} {
		for _, tt := range tests {
			if tasEnabled && tt.skipForTAS {
				continue
			}
			name := fmt.Sprintf("%s (TopologyAwareWorkloadScheduling=%v)", tt.name, tasEnabled)
			t.Run(name, func(t *testing.T) {
				featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
					features.TopologyAwareWorkloadScheduling: tasEnabled,
					features.GenericWorkload:                 true,
				})

				logger, ctx := ktesting.NewTestContext(t)

				client := clientsetfake.NewClientset(testNode)
				informerFactory := informers.NewSharedInformerFactory(client, 0)
				queue := internalqueue.NewSchedulingQueue(nil, informerFactory)
				snapshot := internalcache.NewEmptySnapshot()

				placementFeasiblePlugin := &fakePlacementFeasiblePlugin{
					placementFeasibleStatuses: [][]fwk.Code{tt.podGroupFeasibleStatuses},
				}

				registry := []tf.RegisterPluginFunc{
					tf.RegisterFilterPlugin(tt.plugin.Name(), func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
						return tt.plugin, nil
					}),
					tf.RegisterPostFilterPlugin(tt.plugin.Name(), func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
						return tt.plugin, nil
					}),
					tf.RegisterPermitPlugin(placementFeasiblePlugin.Name(), func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
						return placementFeasiblePlugin, nil
					}),
				}
				schedFwk, err := tf.NewFramework(ctx,
					append(registry,
						tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
						tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
					),
					"test-scheduler",
					frameworkruntime.WithClientSet(client),
					frameworkruntime.WithEventRecorder(events.NewFakeRecorder(100)),
					frameworkruntime.WithInformerFactory(informerFactory),
					frameworkruntime.WithSnapshotSharedLister(snapshot),
					frameworkruntime.WithPodNominator(queue),
				)
				if err != nil {
					t.Fatalf("Failed to create new framework: %v", err)
				}

				cache := internalcache.New(ctx, nil, true)
				cache.AddNode(logger, testNode)

				sched := &Scheduler{
					Cache:            cache,
					nodeInfoSnapshot: snapshot,
					SchedulingQueue:  queue,
					Profiles:         profile.Map{"test-scheduler": schedFwk},
				}
				sched.SchedulePod = sched.schedulePod

				if err := sched.Cache.UpdateSnapshot(logger, sched.nodeInfoSnapshot); err != nil {
					t.Fatalf("Failed to update snapshot: %v", err)
				}

				result := sched.podGroupSchedulingAlgorithm(ctx, schedFwk, framework.NewCycleState(), podGroupInfo, runAllPostFilters)

				if result.status.Code() != tt.expectedGroupStatusCode {
					t.Errorf("Expected group status code: %v, got: %v", tt.expectedGroupStatusCode, result.status.Code())
				}
				if result.waitingOnPreemption != tt.expectedGroupWaitingOnPreemption {
					t.Errorf("Expected group waiting on preemption: %v, got: %v", tt.expectedGroupWaitingOnPreemption, result.waitingOnPreemption)
				}
				if len(tt.expectedPodStatus) != len(result.podResults) {
					t.Errorf("Expected %d pod results, got %d", len(tt.expectedPodStatus), len(result.podResults))
				}
				for i, podResult := range result.podResults {
					podName := podGroupInfo.QueuedPodInfos[i].Pod.Name
					if expected, ok := tt.expectedPodStatus[podName]; ok {
						if podResult.status.Code() != expected.Code() {
							t.Errorf("Expected pod %s status code: %v, got: %v", podName, expected.Code(), podResult.status.Code())
						}
					} else {
						t.Errorf("Got result for unexpected pod %s: %v", podName, podResult.status.Code())
					}
					if podResult.status.IsSuccess() || podResult.requiresPreemption {
						if podResult.scheduleResult.SuggestedHost != "node1" {
							t.Errorf("Expected pod %s suggested host: node1, got: %v", podName, podResult.scheduleResult.SuggestedHost)
						}
					} else {
						if podResult.scheduleResult.SuggestedHost != "" {
							t.Errorf("Expected pod %s empty suggested host, got: %v", podName, podResult.scheduleResult.SuggestedHost)
						}
					}
					if expected, ok := tt.expectedPreemption[podName]; ok {
						if podResult.requiresPreemption != expected {
							t.Errorf("Expected pod %s requiresPreemption: %v, got: %v", podName, expected, podResult.requiresPreemption)
						}
					}
				}
			})
		}
	}
}

// This is only needed because PlacementFeasiblePlugin mock doesn't know which placement it processes and has to assume the order of placements.
// TODO: Remove this once the PlacementFeasiblePlugin becomes order-independent or another way of ordering placements is introduced.
type orderedPlacementPlugin struct {
	fwk.PlacementGeneratePlugin
}

func (p *orderedPlacementPlugin) Name() string {
	return p.PlacementGeneratePlugin.Name() + "_Ordered"
}

func (p *orderedPlacementPlugin) GeneratePlacements(ctx context.Context, state fwk.PodGroupCycleState, podGroup fwk.PodGroupInfo, parentPlacement *fwk.Placement) (*fwk.GeneratePlacementsResult, *fwk.Status) {
	result, status := p.PlacementGeneratePlugin.GeneratePlacements(ctx, state, podGroup, parentPlacement)
	if status.IsSuccess() && result != nil {
		sort.Slice(result.Placements, func(i, j int) bool {
			return result.Placements[i].Name < result.Placements[j].Name
		})
	}
	return result, status
}

func TestSubmitPodGroupAlgorithmResult(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, true)

	testNode := st.MakeNode().Name("node1").UID("node1").Obj()

	p1 := st.MakePod().Name("p1").Namespace("default").UID("p1").PodGroupName("pg").SchedulerName("test-scheduler").Obj()
	p2 := st.MakePod().Name("p2").Namespace("default").UID("p2").PodGroupName("pg").SchedulerName("test-scheduler").Obj()
	p3 := st.MakePod().Name("p3").Namespace("default").UID("p3").PodGroupName("pg").SchedulerName("test-scheduler").Obj()

	testPodGroup := &schedulingv1alpha3.PodGroup{
		ObjectMeta: metav1.ObjectMeta{Name: "pg", Namespace: "default"},
	}

	tests := []struct {
		name                    string
		existingPodGroup        *schedulingv1alpha3.PodGroup
		algorithmResult         podGroupAlgorithmResult
		expectBound             sets.Set[string]
		expectPreempting        sets.Set[string]
		expectFailed            sets.Set[string]
		expectCondition         *metav1.Condition
		expectPodsInActiveQueue sets.Set[string]
	}{
		{
			name: "All pods feasible",
			algorithmResult: podGroupAlgorithmResult{
				status: nil,
				podResults: []algorithmResult{{
					scheduleResult: ScheduleResult{SuggestedHost: "node1"},
					status:         nil,
				}, {
					scheduleResult: ScheduleResult{SuggestedHost: "node1"},
					status:         nil,
				}, {
					scheduleResult: ScheduleResult{SuggestedHost: "node1"},
					status:         nil,
				}},
			},
			expectBound: sets.New("p1", "p2", "p3"),
			expectCondition: &metav1.Condition{
				Type:   schedulingapi.PodGroupInitiallyScheduled,
				Status: metav1.ConditionTrue,
				Reason: "Scheduled",
			},
		},
		{
			name: "All pods feasible, but podGroup unschedulable",
			algorithmResult: podGroupAlgorithmResult{
				status: fwk.NewStatus(fwk.Unschedulable, "not enough capacity for the gang"),
				podResults: []algorithmResult{{
					scheduleResult: ScheduleResult{SuggestedHost: "node1"},
					status:         nil,
				}, {
					scheduleResult: ScheduleResult{SuggestedHost: "node1"},
					status:         nil,
				}, {
					scheduleResult: ScheduleResult{SuggestedHost: "node1"},
					status:         nil,
				}},
			},
			expectFailed: sets.New("p1", "p2", "p3"),
			expectCondition: &metav1.Condition{
				Type:    schedulingapi.PodGroupInitiallyScheduled,
				Status:  metav1.ConditionFalse,
				Reason:  schedulingapi.PodGroupReasonUnschedulable,
				Message: "not enough capacity for the gang",
			},
		},
		{
			name: "One unschedulable",
			algorithmResult: podGroupAlgorithmResult{
				status: nil,
				podResults: []algorithmResult{{
					scheduleResult: ScheduleResult{SuggestedHost: "node1"},
					status:         nil,
				}, {
					scheduleResult: ScheduleResult{SuggestedHost: "", nominatingInfo: clearNominatedNode},
					status:         fwk.NewStatus(fwk.Unschedulable),
				}, {
					scheduleResult: ScheduleResult{SuggestedHost: "node1"},
					status:         nil,
				}},
			},
			expectBound:  sets.New("p1", "p3"),
			expectFailed: sets.New("p2"),
			expectCondition: &metav1.Condition{
				Type:   schedulingapi.PodGroupInitiallyScheduled,
				Status: metav1.ConditionTrue,
				Reason: "Scheduled",
			},
			expectPodsInActiveQueue: sets.New("p2"),
		},
		{
			name: "All pods require preemption",
			algorithmResult: podGroupAlgorithmResult{
				status:              fwk.NewStatus(fwk.Unschedulable, "waiting for preemption to complete"),
				waitingOnPreemption: true,
				podResults: []algorithmResult{{
					scheduleResult: ScheduleResult{
						SuggestedHost:  "node1",
						nominatingInfo: &fwk.NominatingInfo{NominatedNodeName: "node1", NominatingMode: fwk.ModeOverride},
					},
					status:             fwk.NewStatus(fwk.Unschedulable),
					requiresPreemption: true,
				}, {
					scheduleResult: ScheduleResult{
						SuggestedHost:  "node1",
						nominatingInfo: &fwk.NominatingInfo{NominatedNodeName: "node1", NominatingMode: fwk.ModeOverride},
					},
					status:             fwk.NewStatus(fwk.Unschedulable),
					requiresPreemption: true,
				}, {
					scheduleResult: ScheduleResult{
						SuggestedHost:  "node1",
						nominatingInfo: &fwk.NominatingInfo{NominatedNodeName: "node1", NominatingMode: fwk.ModeOverride},
					},
					status:             fwk.NewStatus(fwk.Unschedulable),
					requiresPreemption: true,
				}},
			},
			expectPreempting: sets.New("p1", "p2", "p3"),
			expectCondition: &metav1.Condition{
				Type:    schedulingapi.PodGroupInitiallyScheduled,
				Status:  metav1.ConditionFalse,
				Reason:  schedulingapi.PodGroupReasonUnschedulable,
				Message: "waiting for preemption to complete",
			},
		},
		{
			name: "One pod requires preemption, two are feasible",
			algorithmResult: podGroupAlgorithmResult{
				status:              fwk.NewStatus(fwk.Unschedulable, "waiting for preemption to complete"),
				waitingOnPreemption: true,
				podResults: []algorithmResult{{
					scheduleResult: ScheduleResult{
						SuggestedHost:  "node1",
						nominatingInfo: &fwk.NominatingInfo{NominatedNodeName: "node1", NominatingMode: fwk.ModeOverride},
					},
					status:             fwk.NewStatus(fwk.Unschedulable),
					requiresPreemption: true,
				}, {
					scheduleResult: ScheduleResult{SuggestedHost: "node1"},
					status:         nil,
				}, {
					scheduleResult: ScheduleResult{SuggestedHost: "node1"},
					status:         nil,
				}},
			},
			expectPreempting: sets.New("p1", "p2", "p3"),
			expectCondition: &metav1.Condition{
				Type:    schedulingapi.PodGroupInitiallyScheduled,
				Status:  metav1.ConditionFalse,
				Reason:  schedulingapi.PodGroupReasonUnschedulable,
				Message: "waiting for preemption to complete",
			},
		},
		{
			name: "One pod unschedulable, one requires preemption, one feasible",
			algorithmResult: podGroupAlgorithmResult{
				status:              fwk.NewStatus(fwk.Unschedulable, "waiting for preemption to complete"),
				waitingOnPreemption: true,
				podResults: []algorithmResult{{
					scheduleResult: ScheduleResult{
						SuggestedHost:  "node1",
						nominatingInfo: &fwk.NominatingInfo{NominatedNodeName: "node1", NominatingMode: fwk.ModeOverride},
					},
					status:             fwk.NewStatus(fwk.Unschedulable),
					requiresPreemption: true,
				}, {
					scheduleResult: ScheduleResult{SuggestedHost: "", nominatingInfo: clearNominatedNode},
					status:         fwk.NewStatus(fwk.Unschedulable),
				}, {
					scheduleResult: ScheduleResult{SuggestedHost: "node1"},
					status:         nil,
				}},
			},
			expectPreempting: sets.New("p1", "p3"),
			expectFailed:     sets.New("p2"),
			expectCondition: &metav1.Condition{
				Type:    schedulingapi.PodGroupInitiallyScheduled,
				Status:  metav1.ConditionFalse,
				Reason:  schedulingapi.PodGroupReasonUnschedulable,
				Message: "waiting for preemption to complete",
			},
		},
		{
			name: "All pods unschedulable",
			algorithmResult: podGroupAlgorithmResult{
				status: fwk.NewStatus(fwk.Unschedulable, "0/3 nodes are available: insufficient cpu"),
				podResults: []algorithmResult{{
					scheduleResult: ScheduleResult{SuggestedHost: "", nominatingInfo: clearNominatedNode},
					status:         fwk.NewStatus(fwk.Unschedulable),
				}, {
					scheduleResult: ScheduleResult{SuggestedHost: "", nominatingInfo: clearNominatedNode},
					status:         fwk.NewStatus(fwk.Unschedulable),
				}, {
					scheduleResult: ScheduleResult{SuggestedHost: "", nominatingInfo: clearNominatedNode},
					status:         fwk.NewStatus(fwk.Unschedulable),
				}},
			},
			expectFailed: sets.New("p1", "p2", "p3"),
			expectCondition: &metav1.Condition{
				Type:    schedulingapi.PodGroupInitiallyScheduled,
				Status:  metav1.ConditionFalse,
				Reason:  schedulingapi.PodGroupReasonUnschedulable,
				Message: "0/3 nodes are available: insufficient cpu",
			},
		},
		{
			name: "All pods unschedulable with nil nominatingInfo",
			algorithmResult: podGroupAlgorithmResult{
				status: fwk.NewStatus(fwk.Unschedulable, "0/3 nodes are available: insufficient cpu"),
				podResults: []algorithmResult{{
					scheduleResult: ScheduleResult{SuggestedHost: "", nominatingInfo: nil},
					status:         fwk.NewStatus(fwk.Unschedulable),
				}, {
					scheduleResult: ScheduleResult{SuggestedHost: "", nominatingInfo: nil},
					status:         fwk.NewStatus(fwk.Unschedulable),
				}, {
					scheduleResult: ScheduleResult{SuggestedHost: "", nominatingInfo: nil},
					status:         fwk.NewStatus(fwk.Unschedulable),
				}},
			},
			expectFailed: sets.New("p1", "p2", "p3"),
			expectCondition: &metav1.Condition{
				Type:    schedulingapi.PodGroupInitiallyScheduled,
				Status:  metav1.ConditionFalse,
				Reason:  schedulingapi.PodGroupReasonUnschedulable,
				Message: "0/3 nodes are available: insufficient cpu",
			},
		},
		{
			name: "Unschedulable for the entire pod group",
			algorithmResult: podGroupAlgorithmResult{
				status: fwk.NewStatus(fwk.Unschedulable, "node affinity mismatch"),
				podResults: []algorithmResult{{
					scheduleResult: ScheduleResult{SuggestedHost: "", nominatingInfo: clearNominatedNode},
					status:         fwk.NewStatus(fwk.Unschedulable),
				}, {
					scheduleResult: ScheduleResult{SuggestedHost: "", nominatingInfo: clearNominatedNode},
					status:         fwk.NewStatus(fwk.Unschedulable),
				}, {
					scheduleResult: ScheduleResult{SuggestedHost: "", nominatingInfo: clearNominatedNode},
					status:         fwk.NewStatus(fwk.Unschedulable),
				}},
			},
			expectFailed: sets.New("p1", "p2", "p3"),
			expectCondition: &metav1.Condition{
				Type:    schedulingapi.PodGroupInitiallyScheduled,
				Status:  metav1.ConditionFalse,
				Reason:  schedulingapi.PodGroupReasonUnschedulable,
				Message: "node affinity mismatch",
			},
		},
		{
			name: "Error for one pod",
			algorithmResult: podGroupAlgorithmResult{
				status: fwk.NewStatus(fwk.Error, "plugin returned error"),
				podResults: []algorithmResult{{
					scheduleResult: ScheduleResult{SuggestedHost: "node1"},
					status:         nil,
				}, {
					scheduleResult: ScheduleResult{SuggestedHost: "", nominatingInfo: clearNominatedNode},
					status:         fwk.NewStatus(fwk.Error, "plugin returned error"),
				}, {
					scheduleResult: ScheduleResult{SuggestedHost: "", nominatingInfo: clearNominatedNode},
					status:         fwk.NewStatus(fwk.Error, "plugin returned error"),
				}},
			},
			expectFailed: sets.New("p1", "p2", "p3"),
			expectCondition: &metav1.Condition{
				Type:    schedulingapi.PodGroupInitiallyScheduled,
				Status:  metav1.ConditionFalse,
				Reason:  schedulingapi.PodGroupReasonSchedulerError,
				Message: "plugin returned error",
			},
		},
		{
			name: "Error for one pod while waiting on preemption",
			algorithmResult: podGroupAlgorithmResult{
				status: fwk.NewStatus(fwk.Error, "internal failure"),
				podResults: []algorithmResult{{
					scheduleResult: ScheduleResult{
						SuggestedHost:  "node1",
						nominatingInfo: &fwk.NominatingInfo{NominatedNodeName: "node1", NominatingMode: fwk.ModeOverride},
					},
					status:             fwk.NewStatus(fwk.Unschedulable),
					requiresPreemption: true,
				}, {
					scheduleResult: ScheduleResult{SuggestedHost: "", nominatingInfo: clearNominatedNode},
					status:         fwk.NewStatus(fwk.Error, "internal failure"),
				}, {
					scheduleResult: ScheduleResult{SuggestedHost: "", nominatingInfo: clearNominatedNode},
					status:         fwk.NewStatus(fwk.Error, "internal failure"),
				}},
			},
			expectFailed: sets.New("p1", "p2", "p3"),
			expectCondition: &metav1.Condition{
				Type:    schedulingapi.PodGroupInitiallyScheduled,
				Status:  metav1.ConditionFalse,
				Reason:  schedulingapi.PodGroupReasonSchedulerError,
				Message: "internal failure",
			},
		},
		{
			name: "Already Scheduled, successful cycle keeps condition",
			existingPodGroup: &schedulingv1alpha3.PodGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "pg", Namespace: "default"},
				Status: schedulingv1alpha3.PodGroupStatus{
					Conditions: []metav1.Condition{{
						Type:               schedulingapi.PodGroupInitiallyScheduled,
						Status:             metav1.ConditionTrue,
						Reason:             "Scheduled",
						Message:            "All pods scheduled",
						LastTransitionTime: metav1.Now(),
					}},
				},
			},
			algorithmResult: podGroupAlgorithmResult{
				status: nil,
				podResults: []algorithmResult{{
					scheduleResult: ScheduleResult{SuggestedHost: "node1"},
					status:         nil,
				}, {
					scheduleResult: ScheduleResult{SuggestedHost: "node1"},
					status:         nil,
				}, {
					scheduleResult: ScheduleResult{SuggestedHost: "node1"},
					status:         nil,
				}},
			},
			expectBound: sets.New("p1", "p2", "p3"),
			expectCondition: &metav1.Condition{
				Type:   schedulingapi.PodGroupInitiallyScheduled,
				Status: metav1.ConditionTrue,
				Reason: "Scheduled",
			},
		},
		{
			name: "Already Scheduled, rejected cycle does not regress condition",
			existingPodGroup: &schedulingv1alpha3.PodGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "pg", Namespace: "default"},
				Status: schedulingv1alpha3.PodGroupStatus{
					Conditions: []metav1.Condition{{
						Type:               schedulingapi.PodGroupInitiallyScheduled,
						Status:             metav1.ConditionTrue,
						Reason:             "Scheduled",
						Message:            "All pods scheduled",
						LastTransitionTime: metav1.Now(),
					}},
				},
			},
			algorithmResult: podGroupAlgorithmResult{
				status: fwk.NewStatus(fwk.Unschedulable, "extra pods could not be placed"),
				podResults: []algorithmResult{{
					scheduleResult: ScheduleResult{SuggestedHost: "", nominatingInfo: clearNominatedNode},
					status:         fwk.NewStatus(fwk.Unschedulable),
				}, {
					scheduleResult: ScheduleResult{SuggestedHost: "", nominatingInfo: clearNominatedNode},
					status:         fwk.NewStatus(fwk.Unschedulable),
				}, {
					scheduleResult: ScheduleResult{SuggestedHost: "", nominatingInfo: clearNominatedNode},
					status:         fwk.NewStatus(fwk.Unschedulable),
				}},
			},
			expectFailed: sets.New("p1", "p2", "p3"),
			expectCondition: &metav1.Condition{
				Type:    schedulingapi.PodGroupInitiallyScheduled,
				Status:  metav1.ConditionTrue,
				Reason:  "Scheduled",
				Message: "All pods scheduled",
			},
		},
		{
			name: "Already Scheduled, error cycle does not regress condition",
			existingPodGroup: &schedulingv1alpha3.PodGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "pg", Namespace: "default"},
				Status: schedulingv1alpha3.PodGroupStatus{
					Conditions: []metav1.Condition{{
						Type:               schedulingapi.PodGroupInitiallyScheduled,
						Status:             metav1.ConditionTrue,
						Reason:             "Scheduled",
						Message:            "All pods scheduled",
						LastTransitionTime: metav1.Now(),
					}},
				},
			},
			algorithmResult: podGroupAlgorithmResult{
				status: fwk.NewStatus(fwk.Error),
				podResults: []algorithmResult{{
					scheduleResult: ScheduleResult{SuggestedHost: "node1"},
					status:         nil,
				}, {
					scheduleResult: ScheduleResult{SuggestedHost: "", nominatingInfo: clearNominatedNode},
					status:         fwk.NewStatus(fwk.Error),
				}, {
					scheduleResult: ScheduleResult{SuggestedHost: "", nominatingInfo: clearNominatedNode},
					status:         fwk.NewStatus(fwk.Error),
				}},
			},
			expectFailed: sets.New("p1", "p2", "p3"),
			expectCondition: &metav1.Condition{
				Type:    schedulingapi.PodGroupInitiallyScheduled,
				Status:  metav1.ConditionTrue,
				Reason:  "Scheduled",
				Message: "All pods scheduled",
			},
		},
		{
			name: "Different number of pods in result and queue, should fail all queue pods",
			algorithmResult: podGroupAlgorithmResult{
				status: fwk.NewStatus(fwk.Error),
				podResults: []algorithmResult{{
					scheduleResult: ScheduleResult{SuggestedHost: "node1"},
					status:         nil,
				}},
			},
			expectFailed: sets.New("p1", "p2", "p3"),
			expectCondition: &metav1.Condition{
				Type:    schedulingapi.PodGroupInitiallyScheduled,
				Status:  metav1.ConditionFalse,
				Reason:  schedulingapi.PodGroupReasonSchedulerError,
				Message: fwk.NewStatus(fwk.Error, "scheduling error for pod group, some pods were not processed").AsError().Error(),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)

			var lock sync.Mutex
			boundPods := sets.New[string]()
			preemptingPods := sets.New[string]()
			failedPods := sets.New[string]()

			pg := testPodGroup
			if tt.existingPodGroup != nil {
				pg = tt.existingPodGroup
			}
			client := clientsetfake.NewClientset(testNode, pg)
			client.PrependReactor("create", "pods", func(action clienttesting.Action) (bool, runtime.Object, error) {
				if action.GetSubresource() != "binding" {
					return false, nil, nil
				}
				lock.Lock()
				binding := action.(clienttesting.CreateAction).GetObject().(*v1.Binding)
				boundPods.Insert(binding.Name)
				lock.Unlock()
				return true, binding, nil
			})

			registry := frameworkruntime.Registry{
				queuesort.Name:     queuesort.New,
				defaultbinder.Name: defaultbinder.New,
			}
			profileCfg := config.KubeSchedulerProfile{
				SchedulerName: "test-scheduler",
				Plugins: &config.Plugins{
					QueueSort: config.PluginSet{
						Enabled: []config.Plugin{{Name: queuesort.Name}},
					},
					Bind: config.PluginSet{
						Enabled: []config.Plugin{{Name: defaultbinder.Name}},
					},
				},
			}
			waitingPods := frameworkruntime.NewWaitingPodsMap()
			podInPreBind := frameworkruntime.NewPodsInPreBindMap()
			schedFwk, err := frameworkruntime.NewFramework(ctx, registry, &profileCfg,
				frameworkruntime.WithClientSet(client),
				frameworkruntime.WithEventRecorder(events.NewFakeRecorder(100)),
				frameworkruntime.WithWaitingPods(waitingPods),
				frameworkruntime.WithPodsInPreBind(podInPreBind),
			)
			if err != nil {
				t.Fatalf("Failed to create new framework: %v", err)
			}

			cache := internalcache.New(ctx, nil, true)
			cache.AddNode(logger, testNode)

			informerFactory := informers.NewSharedInformerFactory(client, 0)
			informerFactory.Start(ctx.Done())
			informerFactory.WaitForCacheSync(ctx.Done())

			schedulingQueue := internalqueue.NewTestQueue(ctx, schedFwk.QueueSortFunc())
			sched := &Scheduler{
				client:          client,
				Cache:           cache,
				Profiles:        profile.Map{"test-scheduler": schedFwk},
				SchedulingQueue: schedulingQueue,
				FailureHandler: func(ctx context.Context, fwk framework.Framework, p *framework.QueuedPodInfo, status *fwk.Status, ni *fwk.NominatingInfo, start time.Time) {
					lock.Lock()
					if ni != nil && ni.NominatedNodeName != "" {
						preemptingPods.Insert(p.Pod.Name)
					} else {
						failedPods.Insert(p.Pod.Name)
					}
					lock.Unlock()
					if err := schedulingQueue.AddUnschedulablePodIfNotPresent(logger, p, schedulingQueue.SchedulingCycle()); err != nil {
						t.Fatalf("Unexpected error when adding an unschedulable pod %q to queue: %v", p.Pod.Name, err)
					}
				},
			}

			// Create the pod group and add the pods to queue and pop the group to set up internal queue state correctly.
			schedulingQueue.AddPodGroup(logger, pg)
			schedulingQueue.Add(ctx, p1)
			schedulingQueue.Add(ctx, p2)
			schedulingQueue.Add(ctx, p3)
			entity, err := schedulingQueue.Pop(logger)
			if err != nil {
				t.Fatalf("Failed to pop pod group: %v", err)
			}
			podGroupInfo := entity.(*framework.QueuedPodGroupInfo)
			podGroupInfo.PodGroup = pg
			oldTimestamp := podGroupInfo.Timestamp

			podGroupCycleState := framework.NewCycleState()

			for i := range tt.algorithmResult.podResults {
				pod := podGroupInfo.QueuedPodInfos[i].Pod
				placementCycleState := framework.NewCycleState()
				placementCycleState.SetPodGroupSchedulingCycle(podGroupCycleState)
				podCtx := initPodSchedulingContext(ctx, pod, placementCycleState, runAllPostFilters)
				tt.algorithmResult.podResults[i].podCtx = podCtx
			}

			podGroupPodCount := len(podGroupInfo.QueuedPodInfos)
			sched.submitPodGroupAlgorithmResult(ctx, schedFwk, podGroupCycleState, podGroupInfo, tt.algorithmResult, time.Now())

			if err := wait.PollUntilContextTimeout(ctx, time.Millisecond*200, wait.ForeverTestTimeout, false, func(ctx context.Context) (bool, error) {
				lock.Lock()
				defer lock.Unlock()
				return len(boundPods)+len(preemptingPods)+len(failedPods) == podGroupPodCount, nil
			}); err != nil {
				t.Errorf("Failed waiting for all pods to be either bound or failed")
			}

			if !tt.expectBound.Equal(boundPods) {
				t.Errorf("Expected bound pods: %v, but got: %v", tt.expectBound, boundPods)
			}
			if !tt.expectPreempting.Equal(preemptingPods) {
				t.Errorf("Expected preempting pods: %v, but got: %v", tt.expectPreempting, preemptingPods)
			}
			if !tt.expectFailed.Equal(failedPods) {
				t.Errorf("Expected failed pods: %v, but got: %v", tt.expectFailed, failedPods)
			}

			activePods := sched.SchedulingQueue.PodsInActiveQ()
			activePodNames := sets.New[string]()
			for _, pod := range activePods {
				activePodNames.Insert(pod.Name)
			}
			if diff := cmp.Diff(tt.expectPodsInActiveQueue, activePodNames, cmpopts.EquateEmpty()); diff != "" {
				t.Errorf("Unexpected pods in active queue (-want, +got):\n%s", diff)
			}

			// If there were any remaining pods of the podgroup requeued into the active queue, they must preserve their timestamp.
			if tt.expectPodsInActiveQueue.Len() > 0 {
				queuedPGInfo, ok := sched.SchedulingQueue.GetPodGroup("pg", "default")
				if !ok {
					t.Errorf("Expected pod group pg to be requeued, but it was not found in the scheduling queue")
				} else if !queuedPGInfo.Timestamp.Equal(oldTimestamp) {
					t.Errorf("Expected timestamp to be preserved exactly for pod group. Original: %v, Requeued: %v", oldTimestamp, queuedPGInfo.Timestamp)
				}
			}

			updatedPodGroup, err := client.SchedulingV1alpha3().PodGroups("default").Get(ctx, "pg", metav1.GetOptions{})
			if err != nil {
				t.Fatalf("Failed to get PodGroup: %v", err)
			}
			cond := apimeta.FindStatusCondition(updatedPodGroup.Status.Conditions, schedulingapi.PodGroupInitiallyScheduled)
			if diff := cmp.Diff(tt.expectCondition, cond, cmpopts.IgnoreFields(metav1.Condition{}, "LastTransitionTime")); diff != "" {
				t.Errorf("Unexpected PodGroupInitiallyScheduled condition (-want +got):\n%s", diff)
			}
		})
	}
}

func TestUpdatePodGroupCondition(t *testing.T) {
	tests := []struct {
		name             string
		existingPodGroup *schedulingv1alpha3.PodGroup
		namespace        string
		podGroupName     string
		condition        *metav1.Condition
		expectCondition  *metav1.Condition
		// expectLastTransitionTimeUnchanged, when true, verifies that LastTransitionTime
		// is preserved from the existing condition.
		expectLastTransitionTimeUnchanged bool
	}{
		{
			name: "set Scheduled condition to True on empty status",
			existingPodGroup: &schedulingv1alpha3.PodGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "pg1", Namespace: "ns1"},
			},
			namespace:    "ns1",
			podGroupName: "pg1",
			condition: &metav1.Condition{
				Type:    schedulingapi.PodGroupInitiallyScheduled,
				Status:  metav1.ConditionTrue,
				Reason:  "SomeReason",
				Message: "All required pods have been successfully scheduled",
			},
			expectCondition: &metav1.Condition{
				Type:    schedulingapi.PodGroupInitiallyScheduled,
				Status:  metav1.ConditionTrue,
				Reason:  "SomeReason",
				Message: "All required pods have been successfully scheduled",
			},
		},
		{
			name: "set Scheduled condition to False with Unschedulable reason",
			existingPodGroup: &schedulingv1alpha3.PodGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "pg2", Namespace: "ns1"},
			},
			namespace:    "ns1",
			podGroupName: "pg2",
			condition: &metav1.Condition{
				Type:    schedulingapi.PodGroupInitiallyScheduled,
				Status:  metav1.ConditionFalse,
				Reason:  schedulingapi.PodGroupReasonUnschedulable,
				Message: "0/3 nodes are available: insufficient cpu",
			},
			expectCondition: &metav1.Condition{
				Type:    schedulingapi.PodGroupInitiallyScheduled,
				Status:  metav1.ConditionFalse,
				Reason:  schedulingapi.PodGroupReasonUnschedulable,
				Message: "0/3 nodes are available: insufficient cpu",
			},
		},
		{
			name: "set Scheduled condition to False with SchedulerError reason",
			existingPodGroup: &schedulingv1alpha3.PodGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "pg3", Namespace: "ns1"},
			},
			namespace:    "ns1",
			podGroupName: "pg3",
			condition: &metav1.Condition{
				Type:    schedulingapi.PodGroupInitiallyScheduled,
				Status:  metav1.ConditionFalse,
				Reason:  schedulingapi.PodGroupReasonSchedulerError,
				Message: "Internal scheduling error",
			},
			expectCondition: &metav1.Condition{
				Type:    schedulingapi.PodGroupInitiallyScheduled,
				Status:  metav1.ConditionFalse,
				Reason:  schedulingapi.PodGroupReasonSchedulerError,
				Message: "Internal scheduling error",
			},
		},
		{
			name: "transition from Unschedulable to Scheduled",
			existingPodGroup: &schedulingv1alpha3.PodGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "pg4", Namespace: "ns1"},
				Status: schedulingv1alpha3.PodGroupStatus{
					Conditions: []metav1.Condition{
						{
							Type:               schedulingapi.PodGroupInitiallyScheduled,
							Status:             metav1.ConditionFalse,
							Reason:             schedulingapi.PodGroupReasonUnschedulable,
							Message:            "previously unschedulable",
							LastTransitionTime: metav1.Now(),
						},
					},
				},
			},
			namespace:    "ns1",
			podGroupName: "pg4",
			condition: &metav1.Condition{
				Type:    schedulingapi.PodGroupInitiallyScheduled,
				Status:  metav1.ConditionTrue,
				Reason:  "Scheduled",
				Message: "All required pods have been successfully scheduled",
			},
			expectCondition: &metav1.Condition{
				Type:    schedulingapi.PodGroupInitiallyScheduled,
				Status:  metav1.ConditionTrue,
				Reason:  "Scheduled",
				Message: "All required pods have been successfully scheduled",
			},
		},
		{
			name: "transition from SchedulerError to Scheduled",
			existingPodGroup: &schedulingv1alpha3.PodGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "pg-se-to-true", Namespace: "ns1"},
				Status: schedulingv1alpha3.PodGroupStatus{
					Conditions: []metav1.Condition{
						{
							Type:               schedulingapi.PodGroupInitiallyScheduled,
							Status:             metav1.ConditionFalse,
							Reason:             schedulingapi.PodGroupReasonSchedulerError,
							Message:            "internal error",
							LastTransitionTime: metav1.Now(),
						},
					},
				},
			},
			namespace:    "ns1",
			podGroupName: "pg-se-to-true",
			condition: &metav1.Condition{
				Type:    schedulingapi.PodGroupInitiallyScheduled,
				Status:  metav1.ConditionTrue,
				Reason:  "Scheduled",
				Message: "All required pods have been successfully scheduled",
			},
			expectCondition: &metav1.Condition{
				Type:    schedulingapi.PodGroupInitiallyScheduled,
				Status:  metav1.ConditionTrue,
				Reason:  "Scheduled",
				Message: "All required pods have been successfully scheduled",
			},
		},
		{
			name: "do not regress Scheduled to Unschedulable",
			existingPodGroup: &schedulingv1alpha3.PodGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "pg-true-to-unsched", Namespace: "ns1"},
				Status: schedulingv1alpha3.PodGroupStatus{
					Conditions: []metav1.Condition{
						{
							Type:               schedulingapi.PodGroupInitiallyScheduled,
							Status:             metav1.ConditionTrue,
							Reason:             "Scheduled",
							Message:            "All pods scheduled",
							LastTransitionTime: metav1.Now(),
						},
					},
				},
			},
			namespace:    "ns1",
			podGroupName: "pg-true-to-unsched",
			condition: &metav1.Condition{
				Type:    schedulingapi.PodGroupInitiallyScheduled,
				Status:  metav1.ConditionFalse,
				Reason:  schedulingapi.PodGroupReasonUnschedulable,
				Message: "extra pods could not be placed",
			},
			expectCondition: &metav1.Condition{
				Type:    schedulingapi.PodGroupInitiallyScheduled,
				Status:  metav1.ConditionTrue,
				Reason:  "Scheduled",
				Message: "All pods scheduled",
			},
			expectLastTransitionTimeUnchanged: true,
		},
		{
			name: "do not regress Scheduled to SchedulerError",
			existingPodGroup: &schedulingv1alpha3.PodGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "pg-true-to-se", Namespace: "ns1"},
				Status: schedulingv1alpha3.PodGroupStatus{
					Conditions: []metav1.Condition{
						{
							Type:               schedulingapi.PodGroupInitiallyScheduled,
							Status:             metav1.ConditionTrue,
							Reason:             "Scheduled",
							Message:            "All pods scheduled",
							LastTransitionTime: metav1.Now(),
						},
					},
				},
			},
			namespace:    "ns1",
			podGroupName: "pg-true-to-se",
			condition: &metav1.Condition{
				Type:    schedulingapi.PodGroupInitiallyScheduled,
				Status:  metav1.ConditionFalse,
				Reason:  schedulingapi.PodGroupReasonSchedulerError,
				Message: "internal error",
			},
			expectCondition: &metav1.Condition{
				Type:    schedulingapi.PodGroupInitiallyScheduled,
				Status:  metav1.ConditionTrue,
				Reason:  "Scheduled",
				Message: "All pods scheduled",
			},
			expectLastTransitionTimeUnchanged: true,
		},
		{
			name: "transition from Unschedulable to SchedulerError preserves LastTransitionTime",
			existingPodGroup: &schedulingv1alpha3.PodGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "pg-unsched-to-se", Namespace: "ns1"},
				Status: schedulingv1alpha3.PodGroupStatus{
					Conditions: []metav1.Condition{
						{
							Type:               schedulingapi.PodGroupInitiallyScheduled,
							Status:             metav1.ConditionFalse,
							Reason:             schedulingapi.PodGroupReasonUnschedulable,
							Message:            "not enough resources",
							LastTransitionTime: metav1.Now(),
						},
					},
				},
			},
			namespace:    "ns1",
			podGroupName: "pg-unsched-to-se",
			condition: &metav1.Condition{
				Type:    schedulingapi.PodGroupInitiallyScheduled,
				Status:  metav1.ConditionFalse,
				Reason:  schedulingapi.PodGroupReasonSchedulerError,
				Message: "internal error",
			},
			expectCondition: &metav1.Condition{
				Type:    schedulingapi.PodGroupInitiallyScheduled,
				Status:  metav1.ConditionFalse,
				Reason:  schedulingapi.PodGroupReasonSchedulerError,
				Message: "internal error",
			},
			expectLastTransitionTimeUnchanged: true,
		},
		{
			name: "transition from SchedulerError to Unschedulable preserves LastTransitionTime",
			existingPodGroup: &schedulingv1alpha3.PodGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "pg-se-to-unsched", Namespace: "ns1"},
				Status: schedulingv1alpha3.PodGroupStatus{
					Conditions: []metav1.Condition{
						{
							Type:               schedulingapi.PodGroupInitiallyScheduled,
							Status:             metav1.ConditionFalse,
							Reason:             schedulingapi.PodGroupReasonSchedulerError,
							Message:            "internal error",
							LastTransitionTime: metav1.Now(),
						},
					},
				},
			},
			namespace:    "ns1",
			podGroupName: "pg-se-to-unsched",
			condition: &metav1.Condition{
				Type:    schedulingapi.PodGroupInitiallyScheduled,
				Status:  metav1.ConditionFalse,
				Reason:  schedulingapi.PodGroupReasonUnschedulable,
				Message: "not enough resources",
			},
			expectCondition: &metav1.Condition{
				Type:    schedulingapi.PodGroupInitiallyScheduled,
				Status:  metav1.ConditionFalse,
				Reason:  schedulingapi.PodGroupReasonUnschedulable,
				Message: "not enough resources",
			},
			expectLastTransitionTimeUnchanged: true,
		},
		{
			name: "Scheduled to Scheduled preserves LastTransitionTime",
			existingPodGroup: &schedulingv1alpha3.PodGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "pg-true-to-true", Namespace: "ns1"},
				Status: schedulingv1alpha3.PodGroupStatus{
					Conditions: []metav1.Condition{
						{
							Type:               schedulingapi.PodGroupInitiallyScheduled,
							Status:             metav1.ConditionTrue,
							Reason:             "Scheduled",
							Message:            "All pods scheduled",
							LastTransitionTime: metav1.Now(),
						},
					},
				},
			},
			namespace:    "ns1",
			podGroupName: "pg-true-to-true",
			condition: &metav1.Condition{
				Type:    schedulingapi.PodGroupInitiallyScheduled,
				Status:  metav1.ConditionTrue,
				Reason:  "Scheduled",
				Message: "New condition message",
			},
			expectCondition: &metav1.Condition{
				Type:    schedulingapi.PodGroupInitiallyScheduled,
				Status:  metav1.ConditionTrue,
				Reason:  "Scheduled",
				Message: "New condition message",
			},
			expectLastTransitionTimeUnchanged: true,
		},
		{
			name: "ObservedGeneration is set from PodGroup generation",
			existingPodGroup: &schedulingv1alpha3.PodGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "pg-gen", Namespace: "ns1", Generation: 7},
			},
			namespace:    "ns1",
			podGroupName: "pg-gen",
			condition: &metav1.Condition{
				Type:    schedulingapi.PodGroupInitiallyScheduled,
				Status:  metav1.ConditionTrue,
				Reason:  "Scheduled",
				Message: "All pods scheduled",
			},
			expectCondition: &metav1.Condition{
				Type:               schedulingapi.PodGroupInitiallyScheduled,
				Status:             metav1.ConditionTrue,
				Reason:             "Scheduled",
				Message:            "All pods scheduled",
				ObservedGeneration: 7,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)

			var objects []runtime.Object
			if tt.existingPodGroup != nil {
				objects = append(objects, tt.existingPodGroup)
			}
			client := clientsetfake.NewClientset(objects...)
			informerFactory := informers.NewSharedInformerFactory(client, 0)
			informerFactory.Start(ctx.Done())
			informerFactory.WaitForCacheSync(ctx.Done())
			sched := &Scheduler{client: client}

			var existingLTT metav1.Time
			if existing := apimeta.FindStatusCondition(tt.existingPodGroup.Status.Conditions, schedulingapi.PodGroupInitiallyScheduled); existing != nil {
				existingLTT = existing.LastTransitionTime
			}

			podGroupInfo := &framework.QueuedPodGroupInfo{
				PodGroupInfo: &framework.PodGroupInfo{
					Namespace: tt.namespace,
					Name:      tt.podGroupName,
					PodGroup:  tt.existingPodGroup,
				},
			}
			sched.updatePodGroupCondition(ctx, podGroupInfo, tt.condition)

			updatedPodGroup, err := client.SchedulingV1alpha3().PodGroups(tt.namespace).Get(ctx, tt.podGroupName, metav1.GetOptions{})
			if err != nil {
				t.Fatalf("Failed to get PodGroup: %v", err)
			}

			cond := apimeta.FindStatusCondition(updatedPodGroup.Status.Conditions, tt.expectCondition.Type)
			if diff := cmp.Diff(tt.expectCondition, cond, cmpopts.IgnoreFields(metav1.Condition{}, "LastTransitionTime")); diff != "" {
				t.Errorf("Unexpected PodGroupInitiallyScheduled condition (-want +got):\n%s", diff)
			}

			if tt.expectLastTransitionTimeUnchanged {
				if !cond.LastTransitionTime.Time.Truncate(time.Second).Equal(existingLTT.Time.Truncate(time.Second)) {
					t.Errorf("Expected LastTransitionTime to be preserved as %v, got %v", existingLTT, cond.LastTransitionTime)
				}
			}
		})
	}
}

// fakePlacementPlugin simulates Filter, PlacementGenerate and PlacementScore behaviors for PodGroup placement scheduling testing.
type fakePlacementPlugin struct {
	name                     string
	filterStatus             map[string]*fwk.Status
	generatePlacementsResult map[string][]string
	generatePlacementsStatus *fwk.Status
	scorePlacementsResult    map[string]int64
	scorePlacementsStatus    map[string]*fwk.Status
}

var _ fwk.FilterPlugin = &fakePlacementPlugin{}
var _ fwk.PlacementGeneratePlugin = &fakePlacementPlugin{}
var _ fwk.PlacementScorePlugin = &fakePlacementPlugin{}

func (mp *fakePlacementPlugin) Name() string { return mp.name }

func (mp *fakePlacementPlugin) Filter(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeInfo fwk.NodeInfo) *fwk.Status {
	return mp.filterStatus[nodeInfo.Node().Name]
}

func (mp *fakePlacementPlugin) PlacementScoreExtensions() fwk.PlacementScoreExtensions {
	return nil
}

func (mp *fakePlacementPlugin) ScorePlacement(ctx context.Context, state fwk.PlacementCycleState, podGroup fwk.PodGroupInfo, placement *fwk.PodGroupAssignments) (int64, *fwk.Status) {
	return mp.scorePlacementsResult[placement.Name], mp.scorePlacementsStatus[placement.Name]
}

func (mp *fakePlacementPlugin) GeneratePlacements(ctx context.Context, state fwk.PodGroupCycleState, podGroup fwk.PodGroupInfo, parentPlacement *fwk.Placement) (*fwk.GeneratePlacementsResult, *fwk.Status) {
	parentNodes := map[string]fwk.NodeInfo{}
	for _, node := range parentPlacement.Nodes {
		parentNodes[node.Node().Name] = node
	}

	placements := make([]*fwk.Placement, 0, len(mp.generatePlacementsResult))
	for placementName, nodeNames := range mp.generatePlacementsResult {
		placement := &fwk.Placement{Name: placementName}
		for _, nodeName := range nodeNames {
			placement.Nodes = append(placement.Nodes, parentNodes[nodeName])
		}
		placements = append(placements, placement)
	}
	return &fwk.GeneratePlacementsResult{Placements: placements}, mp.generatePlacementsStatus
}

var statusCmpOpt = cmp.Comparer(func(s1 *fwk.Status, s2 *fwk.Status) bool {
	if s1 == nil || s2 == nil {
		return s1.IsSuccess() && s2.IsSuccess()
	}
	if s1.Code() == fwk.Error {
		return s1.AsError().Error() == s2.AsError().Error()
	}
	return s1.Code() == s2.Code() && s1.Plugin() == s2.Plugin() && s1.Message() == s2.Message()
})

func TestPodGroupSchedulingPlacementAlgorithm(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.TopologyAwareWorkloadScheduling: true,
		features.GenericWorkload:                 true,
	})

	nodes := []*v1.Node{
		st.MakeNode().Name("node1").Obj(),
		st.MakeNode().Name("node2").Obj(),
	}
	podGroupPodInfo, err := framework.NewPodInfo(st.MakePod().Name("foo").UID("foo").PodGroupName("pg").Obj())
	if err != nil {
		t.Fatalf("Failed to create pod info: %v", err)
	}

	tests := map[string]struct {
		placementPlugin           fakePlacementPlugin
		placementFeasibleStatuses [][]fwk.Code
		expectedResult            podGroupAlgorithmResult
	}{
		"respects higher score of placement1": {
			placementPlugin: fakePlacementPlugin{
				generatePlacementsResult: map[string][]string{
					"placement1": {nodes[0].Name},
					"placement2": {nodes[1].Name},
				},
				scorePlacementsResult: map[string]int64{
					"placement1": 2,
					"placement2": 1,
				},
			},
			expectedResult: podGroupAlgorithmResult{
				podResults: []algorithmResult{
					{
						podInfo: podGroupPodInfo,
						scheduleResult: ScheduleResult{
							SuggestedHost:  nodes[0].Name,
							EvaluatedNodes: 1,
							FeasibleNodes:  1,
						},
					},
				},
				status: nil,
			},
		},
		"respects higher score of placement2": {
			placementPlugin: fakePlacementPlugin{
				generatePlacementsResult: map[string][]string{
					"placement1": {nodes[0].Name},
					"placement2": {nodes[1].Name},
				},
				scorePlacementsResult: map[string]int64{
					"placement1": 1,
					"placement2": 2,
				},
			},
			expectedResult: podGroupAlgorithmResult{
				podResults: []algorithmResult{
					{
						podInfo: podGroupPodInfo,
						scheduleResult: ScheduleResult{
							SuggestedHost:  nodes[1].Name,
							EvaluatedNodes: 1,
							FeasibleNodes:  1,
						},
					},
				},
				status: nil,
			},
		},
		"when no placements are generated, returns unschedulable": {
			placementPlugin: fakePlacementPlugin{
				generatePlacementsResult: map[string][]string{},
			},
			expectedResult: podGroupAlgorithmResult{
				status: fwk.NewStatus(fwk.Unschedulable, "no feasible placements found").WithPlugin("FakePlacementPlugin_Ordered"),
			},
		},
		"when all placements are infeasible, returns unschedulable": {
			placementPlugin: fakePlacementPlugin{
				generatePlacementsResult: map[string][]string{
					"placement1": {nodes[0].Name},
					"placement2": {nodes[1].Name},
				},
				scorePlacementsResult: map[string]int64{
					"placement1": 1,
					"placement2": 2,
				},
				filterStatus: map[string]*fwk.Status{
					nodes[0].Name: fwk.NewStatus(fwk.Unschedulable),
					nodes[1].Name: fwk.NewStatus(fwk.Unschedulable),
				},
			},
			expectedResult: podGroupAlgorithmResult{
				podResults: []algorithmResult{
					{
						podInfo: podGroupPodInfo,
						scheduleResult: ScheduleResult{
							EvaluatedNodes: 0,
							FeasibleNodes:  0,
							nominatingInfo: &fwk.NominatingInfo{NominatingMode: fwk.ModeOverride},
						},
						status: fwk.NewStatus(fwk.Unschedulable, "0/1 nodes are available:"),
					},
				},
				status: fwk.NewStatus(fwk.Unschedulable, "0/2 placements are available, first placement status: pod group is unschedulable"),
			},
		},
		"when all placements are infeasible, but pods are feasible, returns unschedulable": {
			placementPlugin: fakePlacementPlugin{
				generatePlacementsResult: map[string][]string{
					"placement1": {nodes[0].Name},
					"placement2": {nodes[1].Name},
				},
				scorePlacementsResult: map[string]int64{
					"placement1": 1,
					"placement2": 2,
				},
				filterStatus: map[string]*fwk.Status{
					nodes[0].Name: nil,
					nodes[1].Name: nil,
				},
			},
			placementFeasibleStatuses: [][]fwk.Code{
				{fwk.Wait, fwk.Unschedulable},
				{fwk.Wait, fwk.Unschedulable},
			},
			expectedResult: podGroupAlgorithmResult{
				podResults: []algorithmResult{
					{
						podInfo: podGroupPodInfo,
						scheduleResult: ScheduleResult{
							SuggestedHost:  "node1",
							EvaluatedNodes: 1,
							FeasibleNodes:  1,
						},
						status: nil,
					},
				},
				status: fwk.NewStatus(fwk.Unschedulable, "0/2 placements are available, first placement status: injected placementFeasible status"),
			},
		},
		"filters out infeasible placements": {
			placementPlugin: fakePlacementPlugin{
				generatePlacementsResult: map[string][]string{
					"placement1": {nodes[0].Name},
					"placement2": {nodes[1].Name},
				},
				scorePlacementsResult: map[string]int64{
					"placement1": 1,
				},
				filterStatus: map[string]*fwk.Status{
					nodes[1].Name: fwk.NewStatus(fwk.Unschedulable),
				},
			},
			expectedResult: podGroupAlgorithmResult{
				podResults: []algorithmResult{
					{
						podInfo: podGroupPodInfo,
						scheduleResult: ScheduleResult{
							SuggestedHost:  nodes[0].Name,
							EvaluatedNodes: 1,
							FeasibleNodes:  1,
						},
					},
				},
				status: nil,
			},
		},
		"filters out infeasible placements with feasible pods": {
			placementPlugin: fakePlacementPlugin{
				generatePlacementsResult: map[string][]string{
					"placement1": {nodes[0].Name},
					"placement2": {nodes[1].Name},
				},
				scorePlacementsResult: map[string]int64{
					"placement1": 1,
					"placement2": 2,
				},
				filterStatus: map[string]*fwk.Status{
					nodes[1].Name: nil,
				},
			},
			placementFeasibleStatuses: [][]fwk.Code{
				{fwk.Wait, fwk.Success},       // placement1
				{fwk.Wait, fwk.Unschedulable}, // placement2
			},
			expectedResult: podGroupAlgorithmResult{
				podResults: []algorithmResult{
					{
						podInfo: podGroupPodInfo,
						scheduleResult: ScheduleResult{
							SuggestedHost:  nodes[0].Name,
							EvaluatedNodes: 1,
							FeasibleNodes:  1,
						},
					},
				},
				status: nil,
			},
		},
		"when generate plugin fails, returns error": {
			placementPlugin: fakePlacementPlugin{
				generatePlacementsStatus: fwk.NewStatus(fwk.Error, "error for test"),
			},
			expectedResult: podGroupAlgorithmResult{
				status: fwk.NewStatus(fwk.Error, "error for test").WithPlugin("FakePlacementPlugin"),
			},
		},
		"when score plugin fails, returns error": {
			placementPlugin: fakePlacementPlugin{
				generatePlacementsResult: map[string][]string{
					"placement1": {nodes[0].Name},
					"placement2": {nodes[1].Name},
				},
				scorePlacementsResult: map[string]int64{
					"placement1": 1,
				},
				scorePlacementsStatus: map[string]*fwk.Status{
					"placement2": fwk.NewStatus(fwk.Error, "error for test"),
				},
			},
			expectedResult: podGroupAlgorithmResult{
				status: fwk.NewStatus(fwk.Error, "running PlacementScore plugins: plugin \"FakePlacementPlugin\" failed with: error for test").WithPlugin("FakePlacementPlugin"),
			},
		},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)

			informerFactory := informers.NewSharedInformerFactory(clientsetfake.NewClientset(), 0)
			queue := internalqueue.NewSchedulingQueue(nil, informerFactory)

			tt.placementPlugin.name = "FakePlacementPlugin"

			orderedPlacementGeneratePlugin := &orderedPlacementPlugin{&tt.placementPlugin}

			placementFeasiblePlugin := &fakePlacementFeasiblePlugin{
				placementFeasibleStatuses: tt.placementFeasibleStatuses,
			}

			registry := []tf.RegisterPluginFunc{
				tf.RegisterPlacementGeneratePlugin(orderedPlacementGeneratePlugin.Name(), func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
					return orderedPlacementGeneratePlugin, nil
				}),
				tf.RegisterPlacementScorePlugin(tt.placementPlugin.Name(), func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
					return &tt.placementPlugin, nil
				}, 1),
				tf.RegisterFilterPlugin(tt.placementPlugin.Name(), func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
					return &tt.placementPlugin, nil
				}),
				tf.RegisterPermitPlugin(placementFeasiblePlugin.Name(), func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
					return placementFeasiblePlugin, nil
				}),
			}

			snapshot := internalcache.NewEmptySnapshot()

			schedFwk, err := tf.NewFramework(ctx,
				append(registry,
					tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
					tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
				),
				"test-scheduler",
				frameworkruntime.WithInformerFactory(informerFactory),
				frameworkruntime.WithSnapshotSharedLister(snapshot),
				frameworkruntime.WithPodNominator(queue),
			)
			if err != nil {
				t.Fatalf("Failed to create new framework: %v", err)
			}

			cache := internalcache.New(ctx, nil, true)
			for _, node := range nodes {
				cache.AddNode(logger, node)
			}

			sched := &Scheduler{
				Cache:            cache,
				nodeInfoSnapshot: snapshot,
				SchedulingQueue:  queue,
				Profiles:         profile.Map{"test-scheduler": schedFwk},
			}
			sched.SchedulePod = sched.schedulePod

			if err := sched.Cache.UpdateSnapshot(logger, sched.nodeInfoSnapshot); err != nil {
				t.Fatalf("Failed to update snapshot: %v", err)
			}

			pgInfo := &framework.QueuedPodGroupInfo{
				QueuedPodInfos: []*framework.QueuedPodInfo{
					{
						PodInfo: podGroupPodInfo,
					},
				},
				PodGroupInfo: &framework.PodGroupInfo{
					UnscheduledPods: []*v1.Pod{podGroupPodInfo.Pod},
				},
			}

			result := sched.podGroupSchedulingPlacementAlgorithm(ctx, schedFwk, framework.NewCycleState(), pgInfo, runAllPostFilters)

			opts := cmp.Options{
				cmp.AllowUnexported(
					podGroupAlgorithmResult{},
					algorithmResult{},
					ScheduleResult{},
					fwk.Status{},
					framework.PodInfo{}),
				cmpopts.IgnoreFields(podGroupAlgorithmResult{}, "placementCycleState"),
				cmpopts.IgnoreFields(algorithmResult{}, "podCtx", "schedulingDuration"),
				statusCmpOpt,
			}

			if diff := cmp.Diff(tt.expectedResult, result, opts...); diff != "" {
				t.Fatalf("Unexpected algorithm result (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestPodGroupSchedulingPlacementAlgorithm_Scoring(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.TopologyAwareWorkloadScheduling: true,
		features.GenericWorkload:                 true,
	})

	nodes := []*v1.Node{
		st.MakeNode().Name("node1").Obj(),
		st.MakeNode().Name("node2").Obj(),
	}
	placements := map[string][]string{
		"placement1": {nodes[0].Name},
		"placement2": {nodes[1].Name},
	}
	podGroupPod := st.MakePod().Name("foo").UID("foo").PodGroupName("pg").Obj()

	type pluginData struct {
		weight               int32
		scorePlacementResult map[string]int64
		scorePlacementStatus map[string]*fwk.Status
	}

	tests := map[string]struct {
		pluginData        []pluginData
		expectedPlacement string
	}{
		"respects higher score of placement1": {
			pluginData: []pluginData{
				{
					weight: 1,
					scorePlacementResult: map[string]int64{
						"placement1": 50,
						"placement2": 75,
					},
				},
				{
					weight: 2,
					scorePlacementResult: map[string]int64{
						"placement1": 25,
						"placement2": 10,
					},
				},
			},
			expectedPlacement: "placement1",
		},
		"respects higher score of placement2": {
			pluginData: []pluginData{
				{
					weight: 1,
					scorePlacementResult: map[string]int64{
						"placement1": 75,
						"placement2": 50,
					},
				},
				{
					weight: 2,
					scorePlacementResult: map[string]int64{
						"placement1": 10,
						"placement2": 25,
					},
				},
			},
			expectedPlacement: "placement2",
		},
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)

			informerFactory := informers.NewSharedInformerFactory(clientsetfake.NewClientset(), 0)
			queue := internalqueue.NewSchedulingQueue(nil, informerFactory)

			placementPlugin := fakePlacementPlugin{
				name:                     "FakeGeneratorPlugin",
				generatePlacementsResult: placements,
			}

			registry := []tf.RegisterPluginFunc{
				tf.RegisterPlacementGeneratePlugin(placementPlugin.Name(), func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
					return &placementPlugin, nil
				}),
				tf.RegisterFilterPlugin(placementPlugin.Name(), func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
					return &placementPlugin, nil
				}),
			}

			for i, placementScorePluginData := range tt.pluginData {
				plugin := fakePlacementPlugin{
					name:                  fmt.Sprintf("FakeScorePlugin[%d]", i),
					scorePlacementsResult: placementScorePluginData.scorePlacementResult,
					scorePlacementsStatus: placementScorePluginData.scorePlacementStatus,
				}

				registry = append(registry, tf.RegisterPlacementScorePlugin(plugin.Name(), func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
					return &plugin, nil
				}, placementScorePluginData.weight))
			}

			snapshot := internalcache.NewEmptySnapshot()

			schedFwk, err := tf.NewFramework(ctx,
				append(registry,
					tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
					tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
				),
				"test-scheduler",
				frameworkruntime.WithInformerFactory(informerFactory),
				frameworkruntime.WithSnapshotSharedLister(snapshot),
				frameworkruntime.WithPodNominator(queue),
			)
			if err != nil {
				t.Fatalf("Failed to create new framework: %v", err)
			}

			cache := internalcache.New(ctx, nil, true)
			for _, node := range nodes {
				cache.AddNode(logger, node)
			}

			sched := &Scheduler{
				Cache:            cache,
				nodeInfoSnapshot: snapshot,
				SchedulingQueue:  queue,
				Profiles:         profile.Map{"test-scheduler": schedFwk},
			}
			sched.SchedulePod = sched.schedulePod

			if err := sched.Cache.UpdateSnapshot(logger, sched.nodeInfoSnapshot); err != nil {
				t.Fatalf("Failed to update snapshot: %v", err)
			}

			pgInfo := &framework.QueuedPodGroupInfo{
				QueuedPodInfos: []*framework.QueuedPodInfo{
					{
						PodInfo: &framework.PodInfo{Pod: podGroupPod},
					},
				},
				PodGroupInfo: &framework.PodGroupInfo{
					UnscheduledPods: []*v1.Pod{podGroupPod},
				},
			}

			result := sched.podGroupSchedulingPlacementAlgorithm(ctx, schedFwk, framework.NewCycleState(), pgInfo, runAllPostFilters)

			expectedHost := placements[tt.expectedPlacement][0]
			actualHost := result.podResults[0].scheduleResult.SuggestedHost
			if expectedHost != actualHost {
				t.Fatalf("Unexpected algorithm result, expected placement %s with node %s, got node %s", tt.expectedPlacement, expectedHost, actualHost)
			}
		})
	}
}

// placementStateTracker is a fake plugin that writes to PlacementCycleState during Filter
// and reads from it during ScorePlacement, to verify the lifecycle of PlacementCycleState.
type placementStateTracker struct {
	name string
	mu   sync.Mutex
	// scoreReadValues records what value was read from PlacementCycleState
	// during each ScorePlacement call, keyed by placement name.
	scoreReadValues map[string]string
	// generatePlacementsResult defines the placements to generate.
	generatePlacementsResult map[string][]string
}

type placementStateData struct {
	value string
}

func (d *placementStateData) Clone() fwk.StateData { return d }

var placementStateKey fwk.StateKey = "placementStateTracker"

var _ fwk.FilterPlugin = &placementStateTracker{}
var _ fwk.PlacementGeneratePlugin = &placementStateTracker{}
var _ fwk.PlacementScorePlugin = &placementStateTracker{}

func (p *placementStateTracker) Name() string { return p.name }

func (p *placementStateTracker) Filter(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeInfo fwk.NodeInfo) *fwk.Status {
	placementState := state.GetPlacementCycleState()
	if placementState == nil {
		return fwk.NewStatus(fwk.Error, "PlacementCycleState is nil during Filter")
	}

	// Write the node name as a marker so ScorePlacement can verify
	// which placement's state it received.
	placementState.Write(placementStateKey, &placementStateData{value: nodeInfo.Node().Name})
	return nil
}

func (p *placementStateTracker) ScorePlacement(ctx context.Context, state fwk.PlacementCycleState, podGroup fwk.PodGroupInfo, placement *fwk.PodGroupAssignments) (int64, *fwk.Status) {
	if state == nil {
		return 0, fwk.NewStatus(fwk.Error, "PlacementCycleState is nil during ScorePlacement")
	}

	data, err := state.Read(placementStateKey)
	if err != nil {
		return 0, fwk.NewStatus(fwk.Error, fmt.Sprintf("failed to read PlacementCycleState for %s: %v", placement.Name, err))
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	p.scoreReadValues[placement.Name] = data.(*placementStateData).value
	return 1, nil
}

func (p *placementStateTracker) PlacementScoreExtensions() fwk.PlacementScoreExtensions {
	return nil
}

func (p *placementStateTracker) GeneratePlacements(ctx context.Context, state fwk.PodGroupCycleState, podGroup fwk.PodGroupInfo, parentPlacement *fwk.Placement) (*fwk.GeneratePlacementsResult, *fwk.Status) {
	parentNodes := map[string]fwk.NodeInfo{}
	for _, node := range parentPlacement.Nodes {
		parentNodes[node.Node().Name] = node
	}

	resultPlacements := make([]*fwk.Placement, 0, len(p.generatePlacementsResult))
	for placementName, nodeNames := range p.generatePlacementsResult {
		placement := &fwk.Placement{Name: placementName}
		for _, nodeName := range nodeNames {
			placement.Nodes = append(placement.Nodes, parentNodes[nodeName])
		}
		resultPlacements = append(resultPlacements, placement)
	}
	return &fwk.GeneratePlacementsResult{Placements: resultPlacements}, nil
}

func TestPlacementCycleStateLifecycle(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.TopologyAwareWorkloadScheduling: true,
		features.GenericWorkload:                 true,
	})

	// A single scenario exercises both isolation and continuity:
	// - Filter writes a node-name marker into PlacementCycleState during each placement's simulation.
	// - ScorePlacement reads from the placement state after all simulations.
	// Assertions verify:
	//   1. Each placement's scorer reads only the value its own simulation wrote (isolation).
	//   2. Data written during each placement's simulation remains readable during its scoring (continuity from simulation to scoring).

	nodes := []*v1.Node{
		st.MakeNode().Name("node1").Obj(),
		st.MakeNode().Name("node2").Obj(),
	}
	podGroupPod := st.MakePod().Name("foo").UID("foo").PodGroupName("pg").Obj()

	logger, ctx := ktesting.NewTestContext(t)

	informerFactory := informers.NewSharedInformerFactory(clientsetfake.NewClientset(), 0)
	queue := internalqueue.NewSchedulingQueue(nil, informerFactory)

	tracker := &placementStateTracker{
		name:            "StateTracker",
		scoreReadValues: make(map[string]string),
		generatePlacementsResult: map[string][]string{
			"placementA": {nodes[0].Name},
			"placementB": {nodes[1].Name},
		},
	}

	registry := []tf.RegisterPluginFunc{
		tf.RegisterPlacementGeneratePlugin(tracker.Name(), func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
			return tracker, nil
		}),
		tf.RegisterPlacementScorePlugin(tracker.Name(), func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
			return tracker, nil
		}, 1),
		tf.RegisterFilterPlugin(tracker.Name(), func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
			return tracker, nil
		}),
	}

	snapshot := internalcache.NewEmptySnapshot()
	schedFwk, err := tf.NewFramework(ctx,
		append(registry,
			tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
			tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
		),
		"test-scheduler",
		frameworkruntime.WithInformerFactory(informerFactory),
		frameworkruntime.WithSnapshotSharedLister(snapshot),
		frameworkruntime.WithPodNominator(queue),
	)
	if err != nil {
		t.Fatalf("Failed to create framework: %v", err)
	}

	cache := internalcache.New(ctx, nil, true)
	for _, node := range nodes {
		cache.AddNode(logger, node)
	}

	sched := &Scheduler{
		Cache:            cache,
		nodeInfoSnapshot: snapshot,
		SchedulingQueue:  queue,
		Profiles:         profile.Map{"test-scheduler": schedFwk},
	}
	sched.SchedulePod = sched.schedulePod

	if err := sched.Cache.UpdateSnapshot(logger, sched.nodeInfoSnapshot); err != nil {
		t.Fatalf("Failed to update snapshot: %v", err)
	}

	pgInfo := &framework.QueuedPodGroupInfo{
		QueuedPodInfos: []*framework.QueuedPodInfo{
			{PodInfo: &framework.PodInfo{Pod: podGroupPod}},
		},
		PodGroupInfo: &framework.PodGroupInfo{
			UnscheduledPods: []*v1.Pod{podGroupPod},
		},
	}

	result := sched.podGroupSchedulingPlacementAlgorithm(ctx, schedFwk, framework.NewCycleState(), pgInfo, runAllPostFilters)
	if !result.status.IsSuccess() {
		t.Fatalf("Expected success, got: %v", result.status)
	}

	// Each placement's scorer must read only what its own simulation wrote
	// (placementA simulated on node1, placementB on node2). This proves both:
	//   - Continuity: data written during a placement's simulation is readable during its scoring.
	//   - Isolation: a placement's scorer does not see another placement's writes.
	expectedScoreReadValues := map[string]string{"placementA": "node1", "placementB": "node2"}
	if diff := cmp.Diff(expectedScoreReadValues, tracker.scoreReadValues); diff != "" {
		t.Errorf("Unexpected scoreReadValues (-want,+got)\n%s", diff)
	}
}

type fakeDefaultPreemption struct {
	*fakePodGroupPlugin
}

func (f *fakeDefaultPreemption) Name() string {
	return names.DefaultPreemption
}

func TestPodGroupCycle_NominatedNodes(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, true)

	testPodGroup := st.MakePodGroup().Name("pg").Namespace("default").Obj()
	p1 := st.MakePod().Name("p1").UID("p1").PodGroupName("pg").SchedulerName("test-scheduler").Obj()
	p2 := st.MakePod().Name("p2").UID("p2").PodGroupName("pg").SchedulerName("test-scheduler").Obj()

	qInfo1 := &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: p1}}
	qInfo2 := &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: p2}}

	podGroupInfo := &framework.QueuedPodGroupInfo{
		QueuedPodInfos: []*framework.QueuedPodInfo{qInfo1, qInfo2},
		PodGroupInfo: &framework.PodGroupInfo{
			Name:            "pg",
			Namespace:       "default",
			UnscheduledPods: []*v1.Pod{p1, p2},
			PodGroup:        testPodGroup,
		},
	}

	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	// Mock PodGroupPostFilter to return NominatingInfos
	nominatedNodes := map[string]*fwk.NominatingInfo{
		p1.Name: {NominatingMode: fwk.ModeOverride, NominatedNodeName: "node1"},
	}
	fakePlugin := &fakePodGroupPlugin{
		podGroupPostFilterStatus: fwk.NewStatus(fwk.Success),
		podGroupPostFilterResult: nominatedNodes,
	}

	registry := frameworkruntime.Registry{
		queuesort.Name:     queuesort.New,
		defaultbinder.Name: defaultbinder.New,
		"DefaultPreemption": func(ctx context.Context, obj runtime.Object, handle fwk.Handle) (fwk.Plugin, error) {
			return &fakeDefaultPreemption{fakePodGroupPlugin: fakePlugin}, nil
		},
	}

	profileCfg := config.KubeSchedulerProfile{
		SchedulerName: "test-scheduler",
		Plugins: &config.Plugins{
			QueueSort: config.PluginSet{
				Enabled: []config.Plugin{{Name: queuesort.Name}},
			},
			Bind: config.PluginSet{
				Enabled: []config.Plugin{{Name: defaultbinder.Name}},
			},
			PostFilter: config.PluginSet{
				Enabled: []config.Plugin{{Name: "DefaultPreemption"}},
			},
			PodGroupPostFilter: config.PluginSet{
				Enabled: []config.Plugin{{Name: "DefaultPreemption"}},
			},
		},
	}

	client := clientsetfake.NewSimpleClientset(testPodGroup)
	informerFactory := informers.NewSharedInformerFactory(client, 0)
	informerFactory.Start(ctx.Done())
	informerFactory.WaitForCacheSync(ctx.Done())

	schedFwk, err := frameworkruntime.NewFramework(ctx, registry, &profileCfg,
		frameworkruntime.WithInformerFactory(informerFactory),
		frameworkruntime.WithClientSet(client),
		frameworkruntime.WithEventRecorder(events.NewFakeRecorder(100)),
	)
	if err != nil {
		t.Fatalf("Failed to create framework: %v", err)
	}

	cache := internalcache.New(ctx, nil, true)
	sched := &Scheduler{
		Profiles:         profile.Map{"test-scheduler": schedFwk},
		Cache:            cache,
		nodeInfoSnapshot: internalcache.NewEmptySnapshot(),
		client:           client,
		SchedulingQueue:  internalqueue.NewTestQueue(ctx, nil),
	}

	// Mock SchedulePod to return Unschedulable initially, and success on subsequent calls
	callCount := 0
	sched.SchedulePod = func(ctx context.Context, fwk framework.Framework, state fwk.CycleState, podInfo *framework.QueuedPodInfo) (ScheduleResult, error) {
		callCount++
		if callCount <= 2 {
			return ScheduleResult{}, &framework.FitError{Pod: podInfo.Pod, NumAllNodes: 1}
		}
		if podInfo.Pod.Name == "p1" {
			return ScheduleResult{SuggestedHost: "node1"}, nil
		}
		if podInfo.Pod.Name == "p2" {
			return ScheduleResult{SuggestedHost: "node2"}, nil
		}
		return ScheduleResult{}, fmt.Errorf("unexpected pod")
	}
	capturedFailureHandler := make(map[string]*fwk.NominatingInfo)
	sched.FailureHandler = func(ctx context.Context, fwk framework.Framework, podInfo *framework.QueuedPodInfo, status *fwk.Status, nominatingInfo *fwk.NominatingInfo, start time.Time) {
		capturedFailureHandler[podInfo.Pod.Name] = nominatingInfo
	}

	// Just inject logger explicitly in context to avoid panic
	logger, _ := ktesting.NewTestContext(t)
	ctx = klog.NewContext(ctx, logger)

	sched.podGroupCycle(ctx, schedFwk, framework.NewCycleState(), podGroupInfo, time.Now())

	if len(capturedFailureHandler) == 0 {
		t.Fatalf("expected FailureHandler to be called")
	}

	if capturedFailureHandler[p1.Name].NominatedNodeName != "node1" {
		t.Errorf("Expected p1 to be nominated for node1, got %s", capturedFailureHandler[p1.Name].NominatedNodeName)
	}

	if capturedFailureHandler[p2.Name] != nil {
		t.Errorf("Expected p2 to not be nominated, got %v", capturedFailureHandler[p2.Name])
	}
}

func TestScheduleOnePodGroup_PodGroupNotFound(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, true)

	p1 := st.MakePod().Name("p1").Namespace("default").UID("p1").PodGroupName("pg").SchedulerName("test-scheduler").Obj()
	p2 := st.MakePod().Name("p2").Namespace("default").UID("p2").PodGroupName("pg").SchedulerName("test-scheduler").Obj()
	qInfo1 := &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: p1}}
	qInfo2 := &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: p2}}

	podGroupInfo := &framework.QueuedPodGroupInfo{
		QueuedPodInfos: []*framework.QueuedPodInfo{qInfo1, qInfo2},
		PodGroupInfo: &framework.PodGroupInfo{
			Name:            "pg",
			Namespace:       "default",
			UnscheduledPods: []*v1.Pod{p1, p2},
		},
	}

	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	registry := frameworkruntime.Registry{
		queuesort.Name:     queuesort.New,
		defaultbinder.Name: defaultbinder.New,
	}
	profileCfg := config.KubeSchedulerProfile{
		SchedulerName: "test-scheduler",
		Plugins: &config.Plugins{
			QueueSort: config.PluginSet{
				Enabled: []config.Plugin{{Name: queuesort.Name}},
			},
			Bind: config.PluginSet{
				Enabled: []config.Plugin{{Name: defaultbinder.Name}},
			},
		},
	}

	client := clientsetfake.NewSimpleClientset(p1, p2)
	informerFactory := informers.NewSharedInformerFactory(client, 0)

	schedFwk, err := frameworkruntime.NewFramework(ctx, registry, &profileCfg,
		frameworkruntime.WithInformerFactory(informerFactory),
		frameworkruntime.WithEventRecorder(events.NewFakeRecorder(100)),
	)
	if err != nil {
		t.Fatalf("Failed to create new framework: %v", err)
	}

	cache := internalcache.New(ctx, nil, true)
	queue := internalqueue.NewSchedulingQueue(nil, informerFactory)

	informerFactory.Start(ctx.Done())
	informerFactory.WaitForCacheSync(ctx.Done())

	sched := &Scheduler{
		Profiles:               profile.Map{"test-scheduler": schedFwk},
		SchedulingQueue:        queue,
		nodeInfoSnapshot:       internalcache.NewEmptySnapshot(),
		Cache:                  cache,
		client:                 client,
		genericWorkloadEnabled: true,
	}
	sched.FailureHandler = sched.handleSchedulingFailure

	sched.scheduleOnePodGroup(ctx, podGroupInfo)

	// Verify that the pods are put back into the scheduling queue's incompletePodGroupPods.
	incompletePods := queue.IncompletePodGroupPodsPods()
	gotIncompletePods := sets.New[string]()
	for _, pod := range incompletePods {
		gotIncompletePods.Insert(pod.Name)
	}
	expectedPods := sets.New("p1", "p2")
	if diff := cmp.Diff(expectedPods, gotIncompletePods); diff != "" {
		t.Errorf("Unexpected pods in incompletePodGroupPods (-want,+got)\n%s", diff)
	}
}

func TestScheduleOnePodGroup_SchedulerNameMismatchUpdatesStatus(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, true)
	p1 := st.MakePod().Name("p1").UID("p1").PodGroupName("pg").SchedulerName("sched1").Obj()
	p2 := st.MakePod().Name("p2").UID("p2").PodGroupName("pg").SchedulerName("sched2").Obj()
	qInfo1 := &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: p1}}
	qInfo2 := &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: p2}}
	testPodGroup := st.MakePodGroup().Name("pg").Namespace("default").Obj()
	podGroupInfo := &framework.QueuedPodGroupInfo{
		QueuedPodInfos: []*framework.QueuedPodInfo{qInfo1, qInfo2},
		PodGroupInfo: &framework.PodGroupInfo{
			Name:      "pg",
			Namespace: "default",
		},
	}
	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	registry := frameworkruntime.Registry{
		queuesort.Name:     queuesort.New,
		defaultbinder.Name: defaultbinder.New,
	}
	profileCfg1 := config.KubeSchedulerProfile{
		SchedulerName: "sched1",
		Plugins: &config.Plugins{
			QueueSort: config.PluginSet{
				Enabled: []config.Plugin{{Name: queuesort.Name}},
			},
			Bind: config.PluginSet{
				Enabled: []config.Plugin{{Name: defaultbinder.Name}},
			},
		},
	}
	schedFwk1, err := frameworkruntime.NewFramework(ctx, registry, &profileCfg1,
		frameworkruntime.WithEventRecorder(events.NewFakeRecorder(100)),
	)
	if err != nil {
		t.Fatalf("Failed to create new framework 1: %v", err)
	}

	profileCfg2 := profileCfg1
	profileCfg2.SchedulerName = "sched2"
	schedFwk2, err := frameworkruntime.NewFramework(ctx, registry, &profileCfg2,
		frameworkruntime.WithEventRecorder(events.NewFakeRecorder(100)),
	)
	if err != nil {
		t.Fatalf("Failed to create new framework 2: %v", err)
	}

	client := clientsetfake.NewClientset(testPodGroup)
	informerFactory := informers.NewSharedInformerFactory(client, 0)
	informerFactory.Start(ctx.Done())
	informerFactory.WaitForCacheSync(ctx.Done())
	sched := &Scheduler{
		Profiles:        profile.Map{"sched1": schedFwk1, "sched2": schedFwk2},
		SchedulingQueue: internalqueue.NewTestQueue(ctx, nil),
		Cache: &fakecache.Cache{
			Cache: internalcache.New(ctx, nil, true),
			UpdateSnapshotFunc: func(nodeSnapshot *internalcache.Snapshot) error {
				return nil
			},
		},
		nodeInfoSnapshot: internalcache.NewTestSnapshotWithPodGroups(
			[]*v1.Pod{st.MakePod().Name("p").Namespace("default").UID("p").PodGroupName("pg").Node("node1").SchedulerName("sched1").Obj()},
			[]*v1.Node{st.MakeNode().Name("node1").Obj()},
			[]*schedulingv1alpha3.PodGroup{st.MakePodGroup().Name("pg").Namespace("default").UID("pg").Obj()},
		),
		client: client,
		FailureHandler: func(ctx context.Context, fwk framework.Framework, p *framework.QueuedPodInfo, status *fwk.Status, ni *fwk.NominatingInfo, start time.Time) {
		},
	}

	sched.scheduleOnePodGroup(ctx, podGroupInfo)

	pg, err := client.SchedulingV1alpha3().PodGroups("default").Get(ctx, "pg", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to get PodGroup: %v", err)
	}
	expectedCondition := metav1.Condition{
		Type:    schedulingapi.PodGroupInitiallyScheduled,
		Status:  metav1.ConditionFalse,
		Reason:  schedulingapi.PodGroupReasonSchedulerError,
		Message: `all pods in a single pod group should have the same .spec.schedulerName set, got: "sched2" and "sched1"`,
	}

	matchedCondition := apimeta.FindStatusCondition(pg.Status.Conditions, schedulingapi.PodGroupInitiallyScheduled)
	if diff := cmp.Diff(&expectedCondition, matchedCondition, cmpopts.IgnoreFields(metav1.Condition{}, "LastTransitionTime", "ObservedGeneration")); diff != "" {
		t.Errorf("Unexpected condition (-want +got):\n%s", diff)
	}
}
