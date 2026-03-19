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
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	v1 "k8s.io/api/core/v1"
	schedulingv1alpha2 "k8s.io/api/scheduling/v1alpha2"
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
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/queuesort"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	"k8s.io/kubernetes/pkg/scheduler/profile"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	tf "k8s.io/kubernetes/pkg/scheduler/testing/framework"
	"k8s.io/utils/ptr"
)

// fakePodGroupPlugin simulates Filter, PostFilter and Permit behaviors for PodGroup scheduling testing.
type fakePodGroupPlugin struct {
	filterStatus     map[string]*fwk.Status
	postFilterResult map[string]*fwk.PostFilterResult
	postFilterStatus map[string]*fwk.Status
	permitStatus     map[string]*fwk.Status
}

var _ fwk.FilterPlugin = &fakePodGroupPlugin{}
var _ fwk.PostFilterPlugin = &fakePodGroupPlugin{}
var _ fwk.PermitPlugin = &fakePodGroupPlugin{}

func (mp *fakePodGroupPlugin) Name() string { return "FakePodGroupPlugin" }

func (mp *fakePodGroupPlugin) Filter(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeInfo fwk.NodeInfo) *fwk.Status {
	if status, ok := mp.filterStatus[pod.Name]; ok {
		return status
	}
	return fwk.NewStatus(fwk.Unschedulable, "default fake filter failure")
}

func (mp *fakePodGroupPlugin) PostFilter(ctx context.Context, state fwk.CycleState, pod *v1.Pod, filteredNodeStatusMap fwk.NodeToStatusReader) (*fwk.PostFilterResult, *fwk.Status) {
	if status, ok := mp.postFilterStatus[pod.Name]; ok {
		return mp.postFilterResult[pod.Name], status
	}
	return &fwk.PostFilterResult{NominatingInfo: clearNominatedNode}, fwk.NewStatus(fwk.Unschedulable, "default fake postfilter failure")
}

func (mp *fakePodGroupPlugin) Permit(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) (*fwk.Status, time.Duration) {
	if status, ok := mp.permitStatus[pod.Name]; ok {
		return status, 0
	}
	return fwk.NewStatus(fwk.Unschedulable, "default fake permit failure"), 0
}

func TestPodGroupInfoForPod(t *testing.T) {
	groupName := "pg"
	p1 := st.MakePod().Name("p1").Namespace("ns1").UID("p1").PodGroupName(groupName).Priority(100).Obj()
	p2 := st.MakePod().Name("p2").Namespace("ns1").UID("p2").PodGroupName(groupName).Priority(200).Obj()
	p3 := st.MakePod().Name("p3").Namespace("ns1").UID("p3").PodGroupName(groupName).Priority(150).Obj()
	p4 := st.MakePod().Name("p4").Namespace("ns1").UID("p4").PodGroupName(groupName).Priority(150).Obj()

	pInfo1, _ := framework.NewPodInfo(p1)
	pInfo2, _ := framework.NewPodInfo(p2)
	pInfo3, _ := framework.NewPodInfo(p3)
	pInfo4, _ := framework.NewPodInfo(p4)

	now := time.Now()
	qInfo1 := &framework.QueuedPodInfo{PodInfo: pInfo1, InitialAttemptTimestamp: ptr.To(now)}
	qInfo2 := &framework.QueuedPodInfo{PodInfo: pInfo2, InitialAttemptTimestamp: ptr.To(now.Add(time.Second))}
	qInfo3 := &framework.QueuedPodInfo{PodInfo: pInfo3, InitialAttemptTimestamp: ptr.To(now.Add(2 * time.Second))}
	qInfo4 := &framework.QueuedPodInfo{PodInfo: pInfo4, InitialAttemptTimestamp: ptr.To(now.Add(time.Second))}

	tests := []struct {
		name            string
		pInfo           *framework.QueuedPodInfo
		unscheduledPods map[string]*v1.Pod
		queuePods       map[types.UID]*framework.QueuedPodInfo
		expectedPods    []string
	}{
		{
			name:  "Success with multiple pods, sorted by priority and timestamp",
			pInfo: qInfo1,
			unscheduledPods: map[string]*v1.Pod{
				"p2": p2,
				"p3": p3,
				"p4": p4,
			},
			queuePods: map[types.UID]*framework.QueuedPodInfo{
				"p2": qInfo2,
				"p3": qInfo3,
				"p4": qInfo4,
			},
			expectedPods: []string{"p2", "p4", "p3", "p1"},
		},
		{
			name:  "Pods not in state or not in queue are skipped",
			pInfo: qInfo1,
			unscheduledPods: map[string]*v1.Pod{
				"p2": p2,
				"p3": p3,
				// p4 missing from unscheduled pods
			},
			queuePods: map[types.UID]*framework.QueuedPodInfo{
				"p2": qInfo2,
				// p3 missing from queue
				"p4": qInfo4,
			},
			expectedPods: []string{"p2", "p1"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			cache := internalcache.New(ctx, nil, true)

			cache.AddPodGroupMember(tt.pInfo.Pod)
			for _, pod := range tt.unscheduledPods {
				cache.AddPodGroupMember(pod)
			}

			q := internalqueue.NewTestQueue(ctx, nil)
			for _, pInfo := range tt.queuePods {
				err := q.AddUnschedulableIfNotPresent(logger, pInfo, 0)
				if err != nil {
					t.Fatalf("Failed to add unschedulable pod: %v", err)
				}
			}
			sched := &Scheduler{
				Cache:           cache,
				SchedulingQueue: q,
			}

			result, err := sched.podGroupInfoForPod(ctx, tt.pInfo)
			if err != nil {
				t.Fatalf("Failed to get pod group info: %v", err)
			}

			if diff := cmp.Diff(groupName, result.PodGroupInfo.Name); diff != "" {
				t.Errorf("Unexpected pod group name (-want,+got):\n%s", diff)
			}

			var gotQueuedPods []string
			for _, pInfo := range result.QueuedPodInfos {
				gotQueuedPods = append(gotQueuedPods, pInfo.Pod.Name)
			}
			if diff := cmp.Diff(tt.expectedPods, gotQueuedPods); diff != "" {
				t.Errorf("Unexpected pod order in QueuedPodInfos (-want,+got):\n%s", diff)
			}

			var gotUnscheduledPods []string
			for _, pod := range result.UnscheduledPods {
				gotUnscheduledPods = append(gotUnscheduledPods, pod.Name)
			}
			if diff := cmp.Diff(tt.expectedPods, gotUnscheduledPods); diff != "" {
				t.Errorf("Unexpected pod order in UnscheduledPods (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestFrameworkForPodGroup(t *testing.T) {
	p1 := st.MakePod().Name("p1").PodGroupName("pg").SchedulerName("sched1").Obj()
	p2 := st.MakePod().Name("p2").PodGroupName("pg").SchedulerName("sched1").Obj()
	p3 := st.MakePod().Name("p3").PodGroupName("pg").SchedulerName("sched2").Obj()

	qInfo1 := &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: p1}}
	qInfo2 := &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: p2}}
	qInfo3 := &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: p3}}

	tests := []struct {
		name        string
		pods        []*framework.QueuedPodInfo
		profiles    profile.Map
		expectError bool
	}{
		{
			name: "success for same scheduler name",
			pods: []*framework.QueuedPodInfo{qInfo1, qInfo2},
			profiles: profile.Map{
				"sched1": nil,
			},
			expectError: false,
		},
		{
			name: "failure for different scheduler names",
			pods: []*framework.QueuedPodInfo{qInfo1, qInfo3},
			profiles: profile.Map{
				"sched1": nil,
				"sched2": nil,
			},
			expectError: true,
		},
		{
			name: "failure when profile not found",
			pods: []*framework.QueuedPodInfo{qInfo1},
			profiles: profile.Map{
				"other": nil,
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sched := &Scheduler{Profiles: tt.profiles}
			podGroupInfo := &framework.QueuedPodGroupInfo{QueuedPodInfos: tt.pods}
			_, err := sched.frameworkForPodGroup(podGroupInfo)
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

	testPodGroup := &schedulingv1alpha2.PodGroup{
		ObjectMeta: metav1.ObjectMeta{Name: "pg", Namespace: "default"},
	}

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
	podGroupLister := informerFactory.Scheduling().V1alpha2().PodGroups().Lister()
	informerFactory.Start(ctx.Done())
	informerFactory.WaitForCacheSync(ctx.Done())

	var failureHandlerCalled bool
	sched := &Scheduler{
		Profiles:        profile.Map{"test-scheduler": schedFwk},
		SchedulingQueue: internalqueue.NewTestQueue(ctx, nil),
		Cache:           cache,
		client:          client,
		podGroupLister:  podGroupLister,
		FailureHandler: func(ctx context.Context, fwk framework.Framework, p *framework.QueuedPodInfo, status *fwk.Status, ni *fwk.NominatingInfo, start time.Time) {
			failureHandlerCalled = true
			if updateSnapshotErr.Error() != status.AsError().Error() {
				t.Errorf("Expected status error %q, got %q", updateSnapshotErr, status.AsError())
			}
		},
	}

	sched.podGroupCycle(ctx, schedFwk, podGroupInfo)

	if !failureHandlerCalled {
		t.Errorf("Expected FailureHandler to be called after UpdateSnapshot failed")
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
		expectedGroupStatusCode          fwk.Code
		expectedGroupWaitingOnPreemption bool
		expectedPodStatus                map[string]*fwk.Status
		expectedPreemption               map[string]bool
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
				permitStatus: map[string]*fwk.Status{
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
			name: "All pods feasible, two waiting",
			plugin: &fakePodGroupPlugin{
				filterStatus: map[string]*fwk.Status{
					"p1": nil,
					"p2": nil,
					"p3": nil,
				},
				permitStatus: map[string]*fwk.Status{
					"p1": fwk.NewStatus(fwk.Wait),
					"p2": fwk.NewStatus(fwk.Wait),
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
			name: "All pods feasible, but all waiting",
			plugin: &fakePodGroupPlugin{
				filterStatus: map[string]*fwk.Status{
					"p1": nil,
					"p2": nil,
					"p3": nil,
				},
				permitStatus: map[string]*fwk.Status{
					"p1": fwk.NewStatus(fwk.Wait),
					"p2": fwk.NewStatus(fwk.Wait),
					"p3": fwk.NewStatus(fwk.Wait),
				},
			},
			expectedGroupStatusCode: fwk.Unschedulable,
			expectedPodStatus: map[string]*fwk.Status{
				"p1": nil,
				"p2": nil,
				"p3": nil,
			},
		},
		{
			name: "All pods feasible, but last waiting",
			plugin: &fakePodGroupPlugin{
				filterStatus: map[string]*fwk.Status{
					"p1": nil,
					"p2": nil,
					"p3": nil,
				},
				permitStatus: map[string]*fwk.Status{
					"p1": nil,
					"p2": nil,
					"p3": fwk.NewStatus(fwk.Wait),
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
			name: "All pods feasible, one waiting, one unschedulable",
			plugin: &fakePodGroupPlugin{
				filterStatus: map[string]*fwk.Status{
					"p1": nil,
					"p2": nil,
					"p3": nil,
				},
				permitStatus: map[string]*fwk.Status{
					"p1": fwk.NewStatus(fwk.Wait),
					"p2": fwk.NewStatus(fwk.Unschedulable),
					"p3": nil,
				},
			},
			expectedGroupStatusCode: fwk.Success,
			expectedPodStatus: map[string]*fwk.Status{
				"p1": nil,
				"p2": fwk.NewStatus(fwk.Unschedulable),
				"p3": nil,
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
				permitStatus: map[string]*fwk.Status{
					"p1": nil,
					"p2": nil,
					"p3": nil,
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
			name: "All pods require preemption, but waiting",
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
				permitStatus: map[string]*fwk.Status{
					"p1": fwk.NewStatus(fwk.Wait),
					"p2": fwk.NewStatus(fwk.Wait),
					"p3": fwk.NewStatus(fwk.Wait),
				},
			},
			expectedGroupStatusCode: fwk.Unschedulable,
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
			// preemption is not yet implemented for TAS
			skipForTAS: true,
		},
		{
			name: "One pod requires preemption, but waiting, two are feasible",
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
				permitStatus: map[string]*fwk.Status{
					"p1": fwk.NewStatus(fwk.Wait),
					"p2": fwk.NewStatus(fwk.Wait),
					"p3": nil,
				},
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
				permitStatus: map[string]*fwk.Status{
					"p1": nil,
					"p3": nil,
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
				permitStatus: map[string]*fwk.Status{
					"p1": nil,
					"p2": nil,
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
			name: "Any permit returned Error",
			plugin: &fakePodGroupPlugin{
				filterStatus: map[string]*fwk.Status{
					"p1": nil,
					"p2": nil,
					"p3": nil,
				},
				permitStatus: map[string]*fwk.Status{
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
			name: "Any filter returned Error while waiting on preemption",
			plugin: &fakePodGroupPlugin{
				filterStatus: map[string]*fwk.Status{
					"p1": fwk.NewStatus(fwk.Unschedulable),
					"p2": fwk.NewStatus(fwk.Error),
					"p3": nil,
				},
				permitStatus: map[string]*fwk.Status{
					"p1": nil,
					"p2": nil,
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
			name: "Any permit returned Error while waiting on preemption",
			plugin: &fakePodGroupPlugin{
				filterStatus: map[string]*fwk.Status{
					"p1": fwk.NewStatus(fwk.Unschedulable),
					"p2": nil,
					"p3": nil,
				},
				permitStatus: map[string]*fwk.Status{
					"p1": nil,
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
	}

	for _, tasEnabled := range []bool{true, false} {
		for _, tt := range tests {
			if tasEnabled && tt.skipForTAS {
				continue
			}
			name := fmt.Sprintf("%s (TopologyAwareWorkloadScheduling=%v)", tt.name, tasEnabled)
			t.Run(name, func(t *testing.T) {
				if tasEnabled {
					featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
						features.TopologyAwareWorkloadScheduling: true,
						features.GenericWorkload:                 true,
					})
				}

				logger, ctx := ktesting.NewTestContext(t)

				client := clientsetfake.NewClientset(testNode)
				informerFactory := informers.NewSharedInformerFactory(client, 0)
				queue := internalqueue.NewSchedulingQueue(nil, informerFactory)
				snapshot := internalcache.NewEmptySnapshot()

				registry := []tf.RegisterPluginFunc{
					tf.RegisterFilterPlugin(tt.plugin.Name(), func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
						return tt.plugin, nil
					}),
					tf.RegisterPostFilterPlugin(tt.plugin.Name(), func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
						return tt.plugin, nil
					}),
					tf.RegisterPermitPlugin(tt.plugin.Name(), func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
						return tt.plugin, nil
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

				result := sched.podGroupSchedulingAlgorithm(ctx, schedFwk, podGroupInfo)

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
						if expected, ok := tt.plugin.permitStatus[podName]; ok {
							if podResult.permitStatus.Code() != expected.Code() {
								t.Errorf("Expected pod %s permit status code: %v, got: %v", podName, expected.Code(), podResult.permitStatus.Code())
							}
						}
					} else {
						if podResult.scheduleResult.SuggestedHost != "" {
							t.Errorf("Expected pod %s empty suggested host, got: %v", podName, podResult.scheduleResult.SuggestedHost)
						}
						if podResult.permitStatus != nil {
							t.Errorf("Expected pod %s nil permit status, got: %v", podName, podResult.permitStatus)
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

func TestSubmitPodGroupAlgorithmResult(t *testing.T) {
	testNode := st.MakeNode().Name("node1").UID("node1").Obj()

	p1 := st.MakePod().Name("p1").UID("p1").PodGroupName("pg").SchedulerName("test-scheduler").Obj()
	p2 := st.MakePod().Name("p2").UID("p2").PodGroupName("pg").SchedulerName("test-scheduler").Obj()
	p3 := st.MakePod().Name("p3").UID("p3").PodGroupName("pg").SchedulerName("test-scheduler").Obj()

	qInfo1 := &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: p1}}
	qInfo2 := &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: p2}}
	qInfo3 := &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: p3}}

	testPodGroup := &schedulingv1alpha2.PodGroup{
		ObjectMeta: metav1.ObjectMeta{Name: "pg", Namespace: "default"},
	}

	podGroupInfo := &framework.QueuedPodGroupInfo{
		QueuedPodInfos: []*framework.QueuedPodInfo{qInfo1, qInfo2, qInfo3},
		PodGroupInfo: &framework.PodGroupInfo{
			Name:            "pg",
			Namespace:       "default",
			UnscheduledPods: []*v1.Pod{p1, p2, p3},
		},
	}

	tests := []struct {
		name             string
		existingPodGroup *schedulingv1alpha2.PodGroup
		algorithmResult  podGroupAlgorithmResult
		expectBound      sets.Set[string]
		expectPreempting sets.Set[string]
		expectFailed     sets.Set[string]
		expectCondition  *metav1.Condition
	}{
		{
			name: "All pods feasible",
			algorithmResult: podGroupAlgorithmResult{
				status: nil,
				podResults: []algorithmResult{{
					scheduleResult: ScheduleResult{SuggestedHost: "node1"},
					status:         nil,
					permitStatus:   nil,
				}, {
					scheduleResult: ScheduleResult{SuggestedHost: "node1"},
					status:         nil,
					permitStatus:   nil,
				}, {
					scheduleResult: ScheduleResult{SuggestedHost: "node1"},
					status:         nil,
					permitStatus:   nil,
				}},
			},
			expectBound: sets.New("p1", "p2", "p3"),
			expectCondition: &metav1.Condition{
				Type:   schedulingapi.PodGroupScheduled,
				Status: metav1.ConditionTrue,
				Reason: "Scheduled",
			},
		},
		{
			name: "All pods feasible, but all waiting",
			algorithmResult: podGroupAlgorithmResult{
				status: fwk.NewStatus(fwk.Unschedulable, "not enough capacity for the gang"),
				podResults: []algorithmResult{{
					scheduleResult: ScheduleResult{SuggestedHost: "node1"},
					status:         nil,
					permitStatus:   fwk.NewStatus(fwk.Wait),
				}, {
					scheduleResult: ScheduleResult{SuggestedHost: "node1"},
					status:         nil,
					permitStatus:   fwk.NewStatus(fwk.Wait),
				}, {
					scheduleResult: ScheduleResult{SuggestedHost: "node1"},
					status:         nil,
					permitStatus:   fwk.NewStatus(fwk.Wait),
				}},
			},
			expectFailed: sets.New("p1", "p2", "p3"),
			expectCondition: &metav1.Condition{
				Type:    schedulingapi.PodGroupScheduled,
				Status:  metav1.ConditionFalse,
				Reason:  schedulingapi.PodGroupReasonUnschedulable,
				Message: "not enough capacity for the gang",
			},
		},
		{
			name: "All pods feasible, but last waiting",
			algorithmResult: podGroupAlgorithmResult{
				status: nil,
				podResults: []algorithmResult{{
					scheduleResult: ScheduleResult{SuggestedHost: "node1"},
					status:         nil,
					permitStatus:   nil,
				}, {
					scheduleResult: ScheduleResult{SuggestedHost: "node1"},
					status:         nil,
					permitStatus:   nil,
				}, {
					scheduleResult: ScheduleResult{SuggestedHost: "node1"},
					status:         nil,
					permitStatus:   fwk.NewStatus(fwk.Wait),
				}},
			},
			expectBound: sets.New("p1", "p2", "p3"),
			expectCondition: &metav1.Condition{
				Type:   schedulingapi.PodGroupScheduled,
				Status: metav1.ConditionTrue,
				Reason: "Scheduled",
			},
		},
		{
			name: "All pods feasible, one waiting, one unschedulable",
			algorithmResult: podGroupAlgorithmResult{
				status: nil,
				podResults: []algorithmResult{{
					scheduleResult: ScheduleResult{SuggestedHost: "node1"},
					status:         nil,
					permitStatus:   fwk.NewStatus(fwk.Wait),
				}, {
					scheduleResult: ScheduleResult{SuggestedHost: "", nominatingInfo: clearNominatedNode},
					status:         fwk.NewStatus(fwk.Unschedulable),
				}, {
					scheduleResult: ScheduleResult{SuggestedHost: "node1"},
					status:         nil,
					permitStatus:   nil,
				}},
			},
			expectBound:  sets.New("p1", "p3"),
			expectFailed: sets.New("p2"),
			expectCondition: &metav1.Condition{
				Type:   schedulingapi.PodGroupScheduled,
				Status: metav1.ConditionTrue,
				Reason: "Scheduled",
			},
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
					permitStatus:       nil,
				}, {
					scheduleResult: ScheduleResult{
						SuggestedHost:  "node1",
						nominatingInfo: &fwk.NominatingInfo{NominatedNodeName: "node1", NominatingMode: fwk.ModeOverride},
					},
					status:             fwk.NewStatus(fwk.Unschedulable),
					requiresPreemption: true,
					permitStatus:       nil,
				}, {
					scheduleResult: ScheduleResult{
						SuggestedHost:  "node1",
						nominatingInfo: &fwk.NominatingInfo{NominatedNodeName: "node1", NominatingMode: fwk.ModeOverride},
					},
					status:             fwk.NewStatus(fwk.Unschedulable),
					requiresPreemption: true,
					permitStatus:       nil,
				}},
			},
			expectPreempting: sets.New("p1", "p2", "p3"),
			expectCondition: &metav1.Condition{
				Type:    schedulingapi.PodGroupScheduled,
				Status:  metav1.ConditionFalse,
				Reason:  schedulingapi.PodGroupReasonUnschedulable,
				Message: "waiting for preemption to complete",
			},
		},
		{
			name: "All pods require preemption, but waiting",
			algorithmResult: podGroupAlgorithmResult{
				status: fwk.NewStatus(fwk.Unschedulable, "preemption required but not feasible"),
				podResults: []algorithmResult{{
					scheduleResult: ScheduleResult{
						SuggestedHost:  "node1",
						nominatingInfo: &fwk.NominatingInfo{NominatedNodeName: "node1", NominatingMode: fwk.ModeOverride},
					},
					status:             fwk.NewStatus(fwk.Unschedulable),
					requiresPreemption: true,
					permitStatus:       fwk.NewStatus(fwk.Wait),
				}, {
					scheduleResult: ScheduleResult{
						SuggestedHost:  "node1",
						nominatingInfo: &fwk.NominatingInfo{NominatedNodeName: "node1", NominatingMode: fwk.ModeOverride},
					},
					status:             fwk.NewStatus(fwk.Unschedulable),
					requiresPreemption: true,
					permitStatus:       fwk.NewStatus(fwk.Wait),
				}, {
					scheduleResult: ScheduleResult{
						SuggestedHost:  "node1",
						nominatingInfo: &fwk.NominatingInfo{NominatedNodeName: "node1", NominatingMode: fwk.ModeOverride},
					},
					status:             fwk.NewStatus(fwk.Unschedulable),
					requiresPreemption: true,
					permitStatus:       fwk.NewStatus(fwk.Wait),
				}},
			},
			expectFailed: sets.New("p1", "p2", "p3"),
			expectCondition: &metav1.Condition{
				Type:    schedulingapi.PodGroupScheduled,
				Status:  metav1.ConditionFalse,
				Reason:  schedulingapi.PodGroupReasonUnschedulable,
				Message: "preemption required but not feasible",
			},
		},
		{
			name: "One pod requires preemption, but waiting, two are feasible",
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
					permitStatus:       fwk.NewStatus(fwk.Wait),
				}, {
					scheduleResult: ScheduleResult{SuggestedHost: "node1"},
					status:         nil,
					permitStatus:   fwk.NewStatus(fwk.Wait),
				}, {
					scheduleResult: ScheduleResult{SuggestedHost: "node1"},
					status:         nil,
					permitStatus:   nil,
				}},
			},
			expectPreempting: sets.New("p1", "p2", "p3"),
			expectCondition: &metav1.Condition{
				Type:    schedulingapi.PodGroupScheduled,
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
					permitStatus:       nil,
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
				Type:    schedulingapi.PodGroupScheduled,
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
			expectBound:  sets.New[string](),
			expectFailed: sets.New("p1", "p2", "p3"),
			expectCondition: &metav1.Condition{
				Type:    schedulingapi.PodGroupScheduled,
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
			expectBound:  sets.New[string](),
			expectFailed: sets.New("p1", "p2", "p3"),
			expectCondition: &metav1.Condition{
				Type:    schedulingapi.PodGroupScheduled,
				Status:  metav1.ConditionFalse,
				Reason:  schedulingapi.PodGroupReasonUnschedulable,
				Message: "0/3 nodes are available: insufficient cpu",
			},
		},
		{
			name: "Unschedulable for the entire pod group",
			algorithmResult: podGroupAlgorithmResult{
				status:     fwk.NewStatus(fwk.Unschedulable, "node affinity mismatch"),
				podResults: []algorithmResult{},
			},
			expectBound:  sets.New[string](),
			expectFailed: sets.New("p1", "p2", "p3"),
			expectCondition: &metav1.Condition{
				Type:    schedulingapi.PodGroupScheduled,
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
					permitStatus:   nil,
				}, {
					scheduleResult: ScheduleResult{SuggestedHost: "", nominatingInfo: clearNominatedNode},
					status:         fwk.NewStatus(fwk.Error, "plugin returned error"),
				}},
			},
			expectBound:  sets.New[string](),
			expectFailed: sets.New("p1", "p2", "p3"),
			expectCondition: &metav1.Condition{
				Type:    schedulingapi.PodGroupScheduled,
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
					permitStatus:       nil,
				}, {
					scheduleResult: ScheduleResult{SuggestedHost: "", nominatingInfo: clearNominatedNode},
					status:         fwk.NewStatus(fwk.Error, "internal failure"),
				}},
			},
			expectBound:  sets.New[string](),
			expectFailed: sets.New("p1", "p2", "p3"),
			expectCondition: &metav1.Condition{
				Type:    schedulingapi.PodGroupScheduled,
				Status:  metav1.ConditionFalse,
				Reason:  schedulingapi.PodGroupReasonSchedulerError,
				Message: "internal failure",
			},
		},
		{
			name: "Already Scheduled, successful cycle keeps condition",
			existingPodGroup: &schedulingv1alpha2.PodGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "pg", Namespace: "default"},
				Status: schedulingv1alpha2.PodGroupStatus{
					Conditions: []metav1.Condition{{
						Type:               schedulingapi.PodGroupScheduled,
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
					permitStatus:   nil,
				}, {
					scheduleResult: ScheduleResult{SuggestedHost: "node1"},
					status:         nil,
					permitStatus:   nil,
				}, {
					scheduleResult: ScheduleResult{SuggestedHost: "node1"},
					status:         nil,
					permitStatus:   nil,
				}},
			},
			expectBound: sets.New("p1", "p2", "p3"),
			expectCondition: &metav1.Condition{
				Type:   schedulingapi.PodGroupScheduled,
				Status: metav1.ConditionTrue,
				Reason: "Scheduled",
			},
		},
		{
			name: "Already Scheduled, rejected cycle does not regress condition",
			existingPodGroup: &schedulingv1alpha2.PodGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "pg", Namespace: "default"},
				Status: schedulingv1alpha2.PodGroupStatus{
					Conditions: []metav1.Condition{{
						Type:               schedulingapi.PodGroupScheduled,
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
				Type:    schedulingapi.PodGroupScheduled,
				Status:  metav1.ConditionTrue,
				Reason:  "Scheduled",
				Message: "All pods scheduled",
			},
		},
		{
			name: "Already Scheduled, error cycle does not regress condition",
			existingPodGroup: &schedulingv1alpha2.PodGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "pg", Namespace: "default"},
				Status: schedulingv1alpha2.PodGroupStatus{
					Conditions: []metav1.Condition{{
						Type:               schedulingapi.PodGroupScheduled,
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
					permitStatus:   nil,
				}, {
					scheduleResult: ScheduleResult{SuggestedHost: "", nominatingInfo: clearNominatedNode},
					status:         fwk.NewStatus(fwk.Error),
				}},
			},
			expectBound:  sets.New[string](),
			expectFailed: sets.New("p1", "p2", "p3"),
			expectCondition: &metav1.Condition{
				Type:    schedulingapi.PodGroupScheduled,
				Status:  metav1.ConditionTrue,
				Reason:  "Scheduled",
				Message: "All pods scheduled",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)

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
			cache.AddNode(klog.FromContext(ctx), testNode)

			informerFactory := informers.NewSharedInformerFactory(client, 0)
			podGroupLister := informerFactory.Scheduling().V1alpha2().PodGroups().Lister()
			informerFactory.Start(ctx.Done())
			informerFactory.WaitForCacheSync(ctx.Done())

			sched := &Scheduler{
				client:          client,
				podGroupLister:  podGroupLister,
				Cache:           cache,
				Profiles:        profile.Map{"test-scheduler": schedFwk},
				SchedulingQueue: internalqueue.NewTestQueue(ctx, nil),
				FailureHandler: func(ctx context.Context, fwk framework.Framework, p *framework.QueuedPodInfo, status *fwk.Status, ni *fwk.NominatingInfo, start time.Time) {
					lock.Lock()
					if ni != nil && ni.NominatedNodeName != "" {
						preemptingPods.Insert(p.Pod.Name)
					} else {
						failedPods.Insert(p.Pod.Name)
					}
					lock.Unlock()
				},
			}

			for i := range tt.algorithmResult.podResults {
				pod := podGroupInfo.QueuedPodInfos[i].Pod
				podCtx := initPodSchedulingContext(ctx, pod)
				tt.algorithmResult.podResults[i].podCtx = podCtx
			}

			sched.submitPodGroupAlgorithmResult(ctx, schedFwk, podGroupInfo, tt.algorithmResult, time.Now())

			if err := wait.PollUntilContextTimeout(ctx, time.Millisecond*200, wait.ForeverTestTimeout, false, func(ctx context.Context) (bool, error) {
				lock.Lock()
				defer lock.Unlock()
				return len(boundPods)+len(preemptingPods)+len(failedPods) == len(podGroupInfo.QueuedPodInfos), nil
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

			updatedPodGroup, err := client.SchedulingV1alpha2().PodGroups("default").Get(ctx, "pg", metav1.GetOptions{})
			if err != nil {
				t.Fatalf("Failed to get PodGroup: %v", err)
			}
			cond := apimeta.FindStatusCondition(updatedPodGroup.Status.Conditions, schedulingapi.PodGroupScheduled)
			if diff := cmp.Diff(tt.expectCondition, cond, cmpopts.IgnoreFields(metav1.Condition{}, "LastTransitionTime")); diff != "" {
				t.Errorf("Unexpected PodGroupScheduled condition (-want +got):\n%s", diff)
			}
		})
	}
}

func TestUpdatePodGroupCondition(t *testing.T) {
	tests := []struct {
		name             string
		existingPodGroup *schedulingv1alpha2.PodGroup
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
			existingPodGroup: &schedulingv1alpha2.PodGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "pg1", Namespace: "ns1"},
			},
			namespace:    "ns1",
			podGroupName: "pg1",
			condition: &metav1.Condition{
				Type:    schedulingapi.PodGroupScheduled,
				Status:  metav1.ConditionTrue,
				Reason:  "SomeReason",
				Message: "All required pods have been successfully scheduled",
			},
			expectCondition: &metav1.Condition{
				Type:    schedulingapi.PodGroupScheduled,
				Status:  metav1.ConditionTrue,
				Reason:  "SomeReason",
				Message: "All required pods have been successfully scheduled",
			},
		},
		{
			name: "set Scheduled condition to False with Unschedulable reason",
			existingPodGroup: &schedulingv1alpha2.PodGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "pg2", Namespace: "ns1"},
			},
			namespace:    "ns1",
			podGroupName: "pg2",
			condition: &metav1.Condition{
				Type:    schedulingapi.PodGroupScheduled,
				Status:  metav1.ConditionFalse,
				Reason:  schedulingapi.PodGroupReasonUnschedulable,
				Message: "0/3 nodes are available: insufficient cpu",
			},
			expectCondition: &metav1.Condition{
				Type:    schedulingapi.PodGroupScheduled,
				Status:  metav1.ConditionFalse,
				Reason:  schedulingapi.PodGroupReasonUnschedulable,
				Message: "0/3 nodes are available: insufficient cpu",
			},
		},
		{
			name: "set Scheduled condition to False with SchedulerError reason",
			existingPodGroup: &schedulingv1alpha2.PodGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "pg3", Namespace: "ns1"},
			},
			namespace:    "ns1",
			podGroupName: "pg3",
			condition: &metav1.Condition{
				Type:    schedulingapi.PodGroupScheduled,
				Status:  metav1.ConditionFalse,
				Reason:  schedulingapi.PodGroupReasonSchedulerError,
				Message: "Internal scheduling error",
			},
			expectCondition: &metav1.Condition{
				Type:    schedulingapi.PodGroupScheduled,
				Status:  metav1.ConditionFalse,
				Reason:  schedulingapi.PodGroupReasonSchedulerError,
				Message: "Internal scheduling error",
			},
		},
		{
			name: "transition from Unschedulable to Scheduled",
			existingPodGroup: &schedulingv1alpha2.PodGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "pg4", Namespace: "ns1"},
				Status: schedulingv1alpha2.PodGroupStatus{
					Conditions: []metav1.Condition{
						{
							Type:               schedulingapi.PodGroupScheduled,
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
				Type:    schedulingapi.PodGroupScheduled,
				Status:  metav1.ConditionTrue,
				Reason:  "Scheduled",
				Message: "All required pods have been successfully scheduled",
			},
			expectCondition: &metav1.Condition{
				Type:    schedulingapi.PodGroupScheduled,
				Status:  metav1.ConditionTrue,
				Reason:  "Scheduled",
				Message: "All required pods have been successfully scheduled",
			},
		},
		{
			name: "transition from SchedulerError to Scheduled",
			existingPodGroup: &schedulingv1alpha2.PodGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "pg-se-to-true", Namespace: "ns1"},
				Status: schedulingv1alpha2.PodGroupStatus{
					Conditions: []metav1.Condition{
						{
							Type:               schedulingapi.PodGroupScheduled,
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
				Type:    schedulingapi.PodGroupScheduled,
				Status:  metav1.ConditionTrue,
				Reason:  "Scheduled",
				Message: "All required pods have been successfully scheduled",
			},
			expectCondition: &metav1.Condition{
				Type:    schedulingapi.PodGroupScheduled,
				Status:  metav1.ConditionTrue,
				Reason:  "Scheduled",
				Message: "All required pods have been successfully scheduled",
			},
		},
		{
			name: "do not regress Scheduled to Unschedulable",
			existingPodGroup: &schedulingv1alpha2.PodGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "pg-true-to-unsched", Namespace: "ns1"},
				Status: schedulingv1alpha2.PodGroupStatus{
					Conditions: []metav1.Condition{
						{
							Type:               schedulingapi.PodGroupScheduled,
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
				Type:    schedulingapi.PodGroupScheduled,
				Status:  metav1.ConditionFalse,
				Reason:  schedulingapi.PodGroupReasonUnschedulable,
				Message: "extra pods could not be placed",
			},
			expectCondition: &metav1.Condition{
				Type:    schedulingapi.PodGroupScheduled,
				Status:  metav1.ConditionTrue,
				Reason:  "Scheduled",
				Message: "All pods scheduled",
			},
			expectLastTransitionTimeUnchanged: true,
		},
		{
			name: "do not regress Scheduled to SchedulerError",
			existingPodGroup: &schedulingv1alpha2.PodGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "pg-true-to-se", Namespace: "ns1"},
				Status: schedulingv1alpha2.PodGroupStatus{
					Conditions: []metav1.Condition{
						{
							Type:               schedulingapi.PodGroupScheduled,
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
				Type:    schedulingapi.PodGroupScheduled,
				Status:  metav1.ConditionFalse,
				Reason:  schedulingapi.PodGroupReasonSchedulerError,
				Message: "internal error",
			},
			expectCondition: &metav1.Condition{
				Type:    schedulingapi.PodGroupScheduled,
				Status:  metav1.ConditionTrue,
				Reason:  "Scheduled",
				Message: "All pods scheduled",
			},
			expectLastTransitionTimeUnchanged: true,
		},
		{
			name: "transition from Unschedulable to SchedulerError preserves LastTransitionTime",
			existingPodGroup: &schedulingv1alpha2.PodGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "pg-unsched-to-se", Namespace: "ns1"},
				Status: schedulingv1alpha2.PodGroupStatus{
					Conditions: []metav1.Condition{
						{
							Type:               schedulingapi.PodGroupScheduled,
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
				Type:    schedulingapi.PodGroupScheduled,
				Status:  metav1.ConditionFalse,
				Reason:  schedulingapi.PodGroupReasonSchedulerError,
				Message: "internal error",
			},
			expectCondition: &metav1.Condition{
				Type:    schedulingapi.PodGroupScheduled,
				Status:  metav1.ConditionFalse,
				Reason:  schedulingapi.PodGroupReasonSchedulerError,
				Message: "internal error",
			},
			expectLastTransitionTimeUnchanged: true,
		},
		{
			name: "transition from SchedulerError to Unschedulable preserves LastTransitionTime",
			existingPodGroup: &schedulingv1alpha2.PodGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "pg-se-to-unsched", Namespace: "ns1"},
				Status: schedulingv1alpha2.PodGroupStatus{
					Conditions: []metav1.Condition{
						{
							Type:               schedulingapi.PodGroupScheduled,
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
				Type:    schedulingapi.PodGroupScheduled,
				Status:  metav1.ConditionFalse,
				Reason:  schedulingapi.PodGroupReasonUnschedulable,
				Message: "not enough resources",
			},
			expectCondition: &metav1.Condition{
				Type:    schedulingapi.PodGroupScheduled,
				Status:  metav1.ConditionFalse,
				Reason:  schedulingapi.PodGroupReasonUnschedulable,
				Message: "not enough resources",
			},
			expectLastTransitionTimeUnchanged: true,
		},
		{
			name: "Scheduled to Scheduled preserves LastTransitionTime",
			existingPodGroup: &schedulingv1alpha2.PodGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "pg-true-to-true", Namespace: "ns1"},
				Status: schedulingv1alpha2.PodGroupStatus{
					Conditions: []metav1.Condition{
						{
							Type:               schedulingapi.PodGroupScheduled,
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
				Type:    schedulingapi.PodGroupScheduled,
				Status:  metav1.ConditionTrue,
				Reason:  "Scheduled",
				Message: "New condition message",
			},
			expectCondition: &metav1.Condition{
				Type:    schedulingapi.PodGroupScheduled,
				Status:  metav1.ConditionTrue,
				Reason:  "Scheduled",
				Message: "New condition message",
			},
			expectLastTransitionTimeUnchanged: true,
		},
		{
			name:         "PodGroup not found does not panic",
			namespace:    "ns1",
			podGroupName: "nonexistent",
			condition: &metav1.Condition{
				Type:    schedulingapi.PodGroupScheduled,
				Status:  metav1.ConditionTrue,
				Reason:  "SomeReason",
				Message: "test",
			},
			expectCondition: &metav1.Condition{
				Type:    schedulingapi.PodGroupScheduled,
				Status:  metav1.ConditionTrue,
				Reason:  "SomeReason",
				Message: "test",
			},
		},
		{
			name: "ObservedGeneration is set from PodGroup generation",
			existingPodGroup: &schedulingv1alpha2.PodGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "pg-gen", Namespace: "ns1", Generation: 7},
			},
			namespace:    "ns1",
			podGroupName: "pg-gen",
			condition: &metav1.Condition{
				Type:    schedulingapi.PodGroupScheduled,
				Status:  metav1.ConditionTrue,
				Reason:  "Scheduled",
				Message: "All pods scheduled",
			},
			expectCondition: &metav1.Condition{
				Type:               schedulingapi.PodGroupScheduled,
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
			podGroupLister := informerFactory.Scheduling().V1alpha2().PodGroups().Lister()
			informerFactory.Start(ctx.Done())
			informerFactory.WaitForCacheSync(ctx.Done())
			sched := &Scheduler{client: client, podGroupLister: podGroupLister}

			var existingLTT metav1.Time
			if tt.existingPodGroup != nil {
				if existing := apimeta.FindStatusCondition(tt.existingPodGroup.Status.Conditions, schedulingapi.PodGroupScheduled); existing != nil {
					existingLTT = existing.LastTransitionTime
				}
			}

			podGroupInfo := &framework.QueuedPodGroupInfo{
				PodGroupInfo: &framework.PodGroupInfo{
					Namespace: tt.namespace,
					Name:      tt.podGroupName,
				},
			}
			sched.updatePodGroupCondition(ctx, podGroupInfo, tt.condition)

			updatedPodGroup, err := client.SchedulingV1alpha2().PodGroups(tt.namespace).Get(ctx, tt.podGroupName, metav1.GetOptions{})
			if tt.existingPodGroup == nil {
				if err == nil {
					t.Fatalf("Expected PodGroup to not be found, but got: %v", updatedPodGroup)
				}
				return
			}
			if err != nil {
				t.Fatalf("Failed to get PodGroup: %v", err)
			}

			cond := apimeta.FindStatusCondition(updatedPodGroup.Status.Conditions, tt.expectCondition.Type)
			if diff := cmp.Diff(tt.expectCondition, cond, cmpopts.IgnoreFields(metav1.Condition{}, "LastTransitionTime")); diff != "" {
				t.Errorf("Unexpected PodGroupScheduled condition (-want +got):\n%s", diff)
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

func (mp *fakePlacementPlugin) ScorePlacement(ctx context.Context, state fwk.PodGroupCycleState, podGroup fwk.PodGroupInfo, placement *fwk.PodGroupAssignments) (int64, *fwk.Status) {
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
	podGroupPod := st.MakePod().Name("foo").UID("foo").PodGroupName("pg").Obj()

	tests := map[string]struct {
		placementPlugin fakePlacementPlugin
		expectedResult  podGroupAlgorithmResult
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
						pod: podGroupPod,
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
						pod: podGroupPod,
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
				status: fwk.NewStatus(fwk.Unschedulable, "no feasible placements found").WithPlugin("FakePlacementPlugin"),
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
						pod: podGroupPod,
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
						pod: podGroupPod,
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

			registry := []tf.RegisterPluginFunc{
				tf.RegisterPlacementGeneratePlugin(tt.placementPlugin.Name(), func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
					return &tt.placementPlugin, nil
				}),
				tf.RegisterPlacementScorePlugin(tt.placementPlugin.Name(), func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
					return &tt.placementPlugin, nil
				}, 1),
				tf.RegisterFilterPlugin(tt.placementPlugin.Name(), func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
					return &tt.placementPlugin, nil
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
						PodInfo: &framework.PodInfo{Pod: podGroupPod},
					},
				},
				PodGroupInfo: &framework.PodGroupInfo{
					UnscheduledPods: []*v1.Pod{podGroupPod},
				},
			}

			result := sched.podGroupSchedulingPlacementAlgorithm(ctx, schedFwk, pgInfo)

			opts := cmp.Options{
				cmp.AllowUnexported(
					podGroupAlgorithmResult{},
					algorithmResult{},
					ScheduleResult{},
					fwk.Status{}),
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

			result := sched.podGroupSchedulingPlacementAlgorithm(ctx, schedFwk, pgInfo)

			expectedHost := placements[tt.expectedPlacement][0]
			actualHost := result.podResults[0].scheduleResult.SuggestedHost
			if expectedHost != actualHost {
				t.Fatalf("Unexpected algorithm result, expected placement %s with node %s, got node %s", tt.expectedPlacement, expectedHost, actualHost)
			}
		})
	}
}
