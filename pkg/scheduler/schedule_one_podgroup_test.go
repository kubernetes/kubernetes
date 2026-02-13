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
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/events"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	internalcache "k8s.io/kubernetes/pkg/scheduler/backend/cache"
	fakecache "k8s.io/kubernetes/pkg/scheduler/backend/cache/fake"
	internalqueue "k8s.io/kubernetes/pkg/scheduler/backend/queue"
	"k8s.io/kubernetes/pkg/scheduler/backend/workloadmanager"
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
	ref := &v1.WorkloadReference{Name: "workload", PodGroup: "pg"}
	p1 := st.MakePod().Name("p1").Namespace("ns1").UID("p1").WorkloadRef(ref).Priority(100).Obj()
	p2 := st.MakePod().Name("p2").Namespace("ns1").UID("p2").WorkloadRef(ref).Priority(200).Obj()
	p3 := st.MakePod().Name("p3").Namespace("ns1").UID("p3").WorkloadRef(ref).Priority(150).Obj()
	p4 := st.MakePod().Name("p4").Namespace("ns1").UID("p4").WorkloadRef(ref).Priority(150).Obj()

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
			name:  "Success with multiple pods, sorted by priority",
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
			name:  "Pod in state but not in queue is skipped",
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
			wm := workloadmanager.New(logger)

			wm.AddPod(tt.pInfo.Pod)
			for _, pod := range tt.unscheduledPods {
				wm.AddPod(pod)
			}

			q := internalqueue.NewTestQueue(ctx, nil)
			for _, pInfo := range tt.queuePods {
				err := q.AddUnschedulableIfNotPresent(logger, pInfo, 0)
				if err != nil {
					t.Fatalf("Failed to add unschedulable pod: %v", err)
				}
			}
			sched := &Scheduler{
				WorkloadManager: wm,
				SchedulingQueue: q,
			}

			result, err := sched.podGroupInfoForPod(ctx, tt.pInfo)
			if err != nil {
				t.Fatalf("Failed to get pod group info: %v", err)
			}

			if diff := cmp.Diff(ref, result.WorkloadRef); diff != "" {
				t.Errorf("Unexpected workload reference (-want,+got):\n%s", diff)
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
	p1 := st.MakePod().Name("p1").SchedulerName("sched1").Obj()
	p2 := st.MakePod().Name("p2").SchedulerName("sched1").Obj()
	p3 := st.MakePod().Name("p3").SchedulerName("sched2").Obj()

	qInfo1 := &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: p1}, NeedsPodGroupCycle: true}
	qInfo2 := &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: p2}, NeedsPodGroupCycle: true}
	qInfo3 := &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: p3}, NeedsPodGroupCycle: true}

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
			pgInfo := &framework.QueuedPodGroupInfo{QueuedPodInfos: tt.pods}
			_, err := sched.frameworkForPodGroup(pgInfo)
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
	p1 := st.MakePod().Name("p1").UID("p1").Obj()
	p2 := st.MakePod().Name("p2").UID("p2").Terminating().Obj()
	p3 := st.MakePod().Name("p3").UID("p3").Obj()

	qInfo1 := &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: p1}, NeedsPodGroupCycle: true}
	qInfo2 := &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: p2}, NeedsPodGroupCycle: true}
	qInfo3 := &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: p3}, NeedsPodGroupCycle: true}

	pgInfo := &framework.QueuedPodGroupInfo{
		QueuedPodInfos: []*framework.QueuedPodInfo{qInfo1, qInfo2, qInfo3},
		PodGroupInfo: &framework.PodGroupInfo{
			UnscheduledPods: []*v1.Pod{p1, p2, p3},
		},
	}

	logger, ctx := ktesting.NewTestContext(t)

	cache := internalcache.New(ctx, nil)
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

	sched.skipPodGroupPodSchedule(ctx, schedFwk, pgInfo)

	if len(pgInfo.QueuedPodInfos) != 1 {
		t.Errorf("Expected 1 queued pod left, got %d", len(pgInfo.QueuedPodInfos))
	}
	if pgInfo.QueuedPodInfos[0].Pod.Name != "p1" {
		t.Errorf("Expected p1 to be left in queued pods, got %s", pgInfo.QueuedPodInfos[0].Pod.Name)
	}
	if len(pgInfo.UnscheduledPods) != 1 {
		t.Errorf("Expected 1 unscheduled pod left, got %d", len(pgInfo.UnscheduledPods))
	}
	if pgInfo.UnscheduledPods[0].Name != "p1" {
		t.Errorf("Expected p1 to be left in unscheduled pods, got %s", pgInfo.UnscheduledPods[0].Name)
	}
}

func TestPodGroupCycle_UpdateSnapshotError(t *testing.T) {
	p1 := st.MakePod().Name("p1").UID("p1").SchedulerName("test-scheduler").Obj()
	p2 := st.MakePod().Name("p2").UID("p2").SchedulerName("test-scheduler").Obj()
	qInfo1 := &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: p1}, NeedsPodGroupCycle: true}
	qInfo2 := &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: p2}, NeedsPodGroupCycle: true}

	pgInfo := &framework.QueuedPodGroupInfo{
		QueuedPodInfos: []*framework.QueuedPodInfo{qInfo1, qInfo2},
		PodGroupInfo: &framework.PodGroupInfo{
			UnscheduledPods: []*v1.Pod{p1, p2},
		},
	}

	_, ctx := ktesting.NewTestContext(t)
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
		Cache: internalcache.New(ctx, nil),
		UpdateSnapshotFunc: func(nodeSnapshot *internalcache.Snapshot) error {
			return updateSnapshotErr
		},
	}

	var failureHandlerCalled bool
	sched := &Scheduler{
		Profiles:        profile.Map{"test-scheduler": schedFwk},
		SchedulingQueue: internalqueue.NewTestQueue(ctx, nil),
		Cache:           cache,
		FailureHandler: func(ctx context.Context, fwk framework.Framework, p *framework.QueuedPodInfo, status *fwk.Status, ni *fwk.NominatingInfo, start time.Time) {
			failureHandlerCalled = true
			if cmp.Equal(updateSnapshotErr.Error(), status.AsError()) {
				t.Errorf("Expected status error %q, got %q", updateSnapshotErr, status.AsError())
			}
		},
	}

	sched.podGroupCycle(ctx, schedFwk, pgInfo, time.Now())

	if !failureHandlerCalled {
		t.Errorf("Expected FailureHandler to be called after UpdateSnapshot failed")
	}
}

func TestPodGroupSchedulingDefaultAlgorithm(t *testing.T) {
	testNode := st.MakeNode().Name("node1").UID("node1").Obj()

	ref := &v1.WorkloadReference{Name: "workload", PodGroup: "pg"}
	p1 := st.MakePod().Name("p1").UID("p1").WorkloadRef(ref).SchedulerName("test-scheduler").Obj()
	p2 := st.MakePod().Name("p2").UID("p2").WorkloadRef(ref).SchedulerName("test-scheduler").Obj()
	p3 := st.MakePod().Name("p3").UID("p3").WorkloadRef(ref).SchedulerName("test-scheduler").Obj()

	qInfo1 := &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: p1}, NeedsPodGroupCycle: true}
	qInfo2 := &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: p2}, NeedsPodGroupCycle: true}
	qInfo3 := &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: p3}, NeedsPodGroupCycle: true}

	pgInfo := &framework.QueuedPodGroupInfo{
		QueuedPodInfos: []*framework.QueuedPodInfo{qInfo1, qInfo2, qInfo3},
		PodGroupInfo: &framework.PodGroupInfo{
			WorkloadRef: &v1.WorkloadReference{
				Name:     "workload",
				PodGroup: "pg",
			},
			UnscheduledPods: []*v1.Pod{p1, p2, p3},
		},
	}

	tests := []struct {
		name                string
		plugin              *fakePodGroupPlugin
		expectedGroupStatus podGroupAlgorithmStatus
		expectedPodStatus   map[string]*fwk.Status
		expectedPreemption  map[string]bool
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
			expectedGroupStatus: podGroupFeasible,
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
			expectedGroupStatus: podGroupFeasible,
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
			expectedGroupStatus: podGroupUnschedulable,
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
			expectedGroupStatus: podGroupFeasible,
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
			expectedGroupStatus: podGroupFeasible,
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
			expectedGroupStatus: podGroupRequiresPreemption,
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
			expectedGroupStatus: podGroupUnschedulable,
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
			expectedGroupStatus: podGroupRequiresPreemption,
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
			expectedGroupStatus: podGroupRequiresPreemption,
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
			expectedGroupStatus: podGroupUnschedulable,
			expectedPodStatus: map[string]*fwk.Status{
				"p1": fwk.NewStatus(fwk.Unschedulable),
				"p2": fwk.NewStatus(fwk.Unschedulable),
				"p3": fwk.NewStatus(fwk.Unschedulable),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)

			client := clientsetfake.NewClientset(testNode)
			informerFactory := informers.NewSharedInformerFactory(client, 0)
			queue := internalqueue.NewSchedulingQueue(nil, informerFactory)

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
				frameworkruntime.WithSnapshotSharedLister(internalcache.NewEmptySnapshot()),
				frameworkruntime.WithPodNominator(queue),
			)
			if err != nil {
				t.Fatalf("Failed to create new framework: %v", err)
			}

			cache := internalcache.New(ctx, nil)
			cache.AddNode(logger, testNode)

			sched := &Scheduler{
				Cache:            cache,
				nodeInfoSnapshot: internalcache.NewEmptySnapshot(),
				SchedulingQueue:  queue,
				Profiles:         profile.Map{"test-scheduler": schedFwk},
			}
			sched.SchedulePod = sched.schedulePod

			if err := sched.Cache.UpdateSnapshot(logger, sched.nodeInfoSnapshot); err != nil {
				t.Fatalf("Failed to update snapshot: %v", err)
			}

			result := sched.podGroupSchedulingDefaultAlgorithm(ctx, schedFwk, pgInfo)

			if result.status != tt.expectedGroupStatus {
				t.Errorf("Expected group status: %v, got: %v", tt.expectedGroupStatus, result.status)
			}
			for i, podResult := range result.podResults {
				podName := pgInfo.QueuedPodInfos[i].Pod.Name
				if expected, ok := tt.expectedPodStatus[podName]; ok {
					if podResult.status.Code() != expected.Code() {
						t.Errorf("Expected pod %s status code: %v, got: %v", podName, expected.Code(), podResult.status.Code())
					}
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

func TestSubmitPodGroupAlgorithmResult(t *testing.T) {
	testNode := st.MakeNode().Name("node1").UID("node1").Obj()

	ref := &v1.WorkloadReference{Name: "workload", PodGroup: "pg"}
	p1 := st.MakePod().Name("p1").UID("p1").WorkloadRef(ref).SchedulerName("test-scheduler").Obj()
	p2 := st.MakePod().Name("p2").UID("p2").WorkloadRef(ref).SchedulerName("test-scheduler").Obj()
	p3 := st.MakePod().Name("p3").UID("p3").WorkloadRef(ref).SchedulerName("test-scheduler").Obj()

	qInfo1 := &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: p1}, NeedsPodGroupCycle: true}
	qInfo2 := &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: p2}, NeedsPodGroupCycle: true}
	qInfo3 := &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: p3}, NeedsPodGroupCycle: true}

	pgInfo := &framework.QueuedPodGroupInfo{
		QueuedPodInfos: []*framework.QueuedPodInfo{qInfo1, qInfo2, qInfo3},
		PodGroupInfo: &framework.PodGroupInfo{
			UnscheduledPods: []*v1.Pod{p1, p2, p3},
		},
	}

	tests := []struct {
		name             string
		algorithmResult  podGroupAlgorithmResult
		expectBound      sets.Set[string]
		expectPreempting sets.Set[string]
		expectFailed     sets.Set[string]
	}{
		{
			name: "All pods feasible",
			algorithmResult: podGroupAlgorithmResult{
				status: podGroupFeasible,
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
		},
		{
			name: "All pods feasible, but all waiting",
			algorithmResult: podGroupAlgorithmResult{
				status: podGroupUnschedulable,
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
		},
		{
			name: "All pods feasible, but last waiting",
			algorithmResult: podGroupAlgorithmResult{
				status: podGroupFeasible,
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
		},
		{
			name: "All pods feasible, one waiting, one unschedulable",
			algorithmResult: podGroupAlgorithmResult{
				status: podGroupFeasible,
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
		},
		{
			name: "All pods require preemption",
			algorithmResult: podGroupAlgorithmResult{
				status: podGroupRequiresPreemption,
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
		},
		{
			name: "All pods require preemption, but waiting",
			algorithmResult: podGroupAlgorithmResult{
				status: podGroupUnschedulable,
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
		},
		{
			name: "One pod requires preemption, but waiting, two are feasible",
			algorithmResult: podGroupAlgorithmResult{
				status: podGroupRequiresPreemption,
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
		},
		{
			name: "One pod unschedulable, one requires preemption, one feasible",
			algorithmResult: podGroupAlgorithmResult{
				status: podGroupRequiresPreemption,
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
		},
		{
			name: "All pods unschedulable",
			algorithmResult: podGroupAlgorithmResult{
				status: podGroupUnschedulable,
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
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)

			var lock sync.Mutex
			boundPods := sets.New[string]()
			preemptingPods := sets.New[string]()
			failedPods := sets.New[string]()

			client := clientsetfake.NewClientset(testNode)
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
			schedFwk, err := frameworkruntime.NewFramework(ctx, registry, &profileCfg,
				frameworkruntime.WithClientSet(client),
				frameworkruntime.WithEventRecorder(events.NewFakeRecorder(100)),
				frameworkruntime.WithWaitingPods(waitingPods),
			)
			if err != nil {
				t.Fatalf("Failed to create new framework: %v", err)
			}

			cache := internalcache.New(ctx, nil)
			cache.AddNode(klog.FromContext(ctx), testNode)

			sched := &Scheduler{
				Cache:           cache,
				Profiles:        profile.Map{"test-scheduler": schedFwk},
				SchedulingQueue: internalqueue.NewTestQueue(ctx, nil),
				FailureHandler: func(ctx context.Context, fwk framework.Framework, p *framework.QueuedPodInfo, status *fwk.Status, ni *fwk.NominatingInfo, start time.Time) {
					lock.Lock()
					if ni.NominatedNodeName != "" {
						preemptingPods.Insert(p.Pod.Name)
					} else {
						failedPods.Insert(p.Pod.Name)
					}
					lock.Unlock()
				},
			}

			for i := range pgInfo.QueuedPodInfos {
				podCtx := sched.initPodSchedulingContext(ctx, p1)
				tt.algorithmResult.podResults[i].podCtx = podCtx
			}

			sched.submitPodGroupAlgorithmResult(ctx, schedFwk, pgInfo, tt.algorithmResult)

			if err := wait.PollUntilContextTimeout(ctx, time.Millisecond*200, wait.ForeverTestTimeout, false, func(ctx context.Context) (bool, error) {
				lock.Lock()
				defer lock.Unlock()
				return len(boundPods)+len(preemptingPods)+len(failedPods) == len(pgInfo.QueuedPodInfos), nil
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
		})
	}
}
