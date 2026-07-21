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
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	v1 "k8s.io/api/core/v1"
	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/resource"
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
	componentmetrics "k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/testutil"
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
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/gangscheduling"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/queuesort"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	"k8s.io/kubernetes/pkg/scheduler/profile"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	tf "k8s.io/kubernetes/pkg/scheduler/testing/framework"
	testingclock "k8s.io/utils/clock/testing"
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

const fakePlacementFeasiblePluginDataKey = "fakePlacementFeasiblePluginDataKey"

type fakePlacementFeasiblePluginData struct {
	placementIndex int
}

func (d *fakePlacementFeasiblePluginData) Clone() fwk.StateData {
	return &fakePlacementFeasiblePluginData{placementIndex: d.placementIndex}
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
func (mp *fakePlacementFeasiblePlugin) PlacementFeasible(ctx context.Context, placementCycleState fwk.PlacementCycleState, podGroupInfo fwk.PodGroupInfo, args framework.PlacementProgress) *fwk.Status {
	// If no mock statuses are configured, always succeed.
	if len(mp.placementFeasibleStatuses) == 0 {
		return nil
	}

	total := len(podGroupInfo.GetUnscheduledPods())
	if pgInfo, ok := podGroupInfo.(*framework.PodGroupInfo); ok && pgInfo.Type == fwk.CompositePodGroupKeyType {
		total = len(pgInfo.Children)
	}
	evaluated := total - args.Remaining

	if evaluated == 0 {
		mp.placementCount++
	}

	state, err := placementCycleState.Read(fakePlacementFeasiblePluginDataKey)
	if err != nil {
		state = &fakePlacementFeasiblePluginData{placementIndex: mp.placementCount - 1}
		placementCycleState.Write(fakePlacementFeasiblePluginDataKey, state)
	}
	placementIndex := state.(*fakePlacementFeasiblePluginData).placementIndex

	// Ensure the indices are within the bounds of the injected statuses.
	if placementIndex < len(mp.placementFeasibleStatuses) {
		// If the specific placement has no pod statuses configured, treat it as always successful.
		if len(mp.placementFeasibleStatuses[placementIndex]) == 0 {
			return nil
		}
		if evaluated < len(mp.placementFeasibleStatuses[placementIndex]) {
			code := mp.placementFeasibleStatuses[placementIndex][evaluated]
			if code == fwk.Success {
				return nil
			}
			return fwk.NewStatus(code, "injected placementFeasible status")
		}
	}
	return nil
}

func (mp *fakePlacementFeasiblePlugin) Permit(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) (*fwk.Status, time.Duration) {
	return fwk.NewStatus(fwk.Error, "unexpected call to permit"), 0
}

func TestValidatePodGroup(t *testing.T) {
	tests := []struct {
		name                           string
		podGroup                       *schedulingv1beta1.PodGroup
		scheduledPods                  []*v1.Pod
		pods                           []*v1.Pod
		profiles                       profile.Map
		expectError                    bool
		enablePodGroupPreemptionPolicy bool
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
			expectError: false,
		},
		{
			name:     "failure when different priorities across pods",
			podGroup: st.MakePodGroup().Name("pg").Priority(10).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Name("p1").PodGroupName("pg").Priority(9).Obj(),
				st.MakePod().Name("p2").PodGroupName("pg").Priority(10).Obj(),
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
		{
			name:     "success when preemption policies match",
			podGroup: st.MakePodGroup().Name("pg").PreemptionPolicy(schedulingv1beta1.PreemptNever).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Name("p1").PodGroupName("pg").PreemptionPolicy(v1.PreemptNever).Obj(),
				st.MakePod().Name("p2").PodGroupName("pg").PreemptionPolicy(v1.PreemptNever).Obj(),
			},
			enablePodGroupPreemptionPolicy: true,
			expectError:                    false,
		},
		{
			name:     "failure when different preemption policies across pods",
			podGroup: st.MakePodGroup().Name("pg").PreemptionPolicy(schedulingv1beta1.PreemptNever).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Name("p1").PodGroupName("pg").PreemptionPolicy(v1.PreemptLowerPriority).Obj(),
				st.MakePod().Name("p2").PodGroupName("pg").PreemptionPolicy(v1.PreemptNever).Obj(),
			},
			enablePodGroupPreemptionPolicy: true,
			expectError:                    true,
		},
		{
			name:     "failure when different preemption policies across pods and pod group",
			podGroup: st.MakePodGroup().Name("pg").PreemptionPolicy(schedulingv1beta1.PreemptNever).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Name("p1").PodGroupName("pg").PreemptionPolicy(v1.PreemptLowerPriority).Obj(),
				st.MakePod().Name("p2").PodGroupName("pg").PreemptionPolicy(v1.PreemptLowerPriority).Obj(),
			},
			enablePodGroupPreemptionPolicy: true,
			expectError:                    true,
		},
		{
			name:     "success when preemption policies between pods and podgroup do not match but PodGroupPreemptionPolicy is disabled",
			podGroup: st.MakePodGroup().Name("pg").PreemptionPolicy(schedulingv1beta1.PreemptNever).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Name("p1").PodGroupName("pg").PreemptionPolicy(v1.PreemptLowerPriority).Obj(),
				st.MakePod().Name("p2").PodGroupName("pg").PreemptionPolicy(v1.PreemptLowerPriority).Obj(),
			},
			enablePodGroupPreemptionPolicy: false,
			expectError:                    false,
		},
		{
			name:     "failure when preemption policies do not match across pods and PodGroupPreemptionPolicy is disabled",
			podGroup: st.MakePodGroup().Name("pg").PreemptionPolicy(schedulingv1beta1.PreemptNever).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Name("p1").PodGroupName("pg").PreemptionPolicy(v1.PreemptLowerPriority).Obj(),
				st.MakePod().Name("p2").PodGroupName("pg").PreemptionPolicy(v1.PreemptNever).Obj(),
			},
			enablePodGroupPreemptionPolicy: false,
			expectError:                    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload:          true,
				features.PodGroupPreemptionPolicy: tt.enablePodGroupPreemptionPolicy,
			})
			snapshot := internalcache.NewTestSnapshotWithPodGroups(tt.scheduledPods, nil, []*schedulingv1beta1.PodGroup{tt.podGroup})
			profilesOrDefault := func(p profile.Map) profile.Map {
				if p == nil {
					return profile.Map{
						"": nil,
					}
				}
				return p
			}
			sched := &Scheduler{
				Profiles:         profilesOrDefault(tt.profiles),
				nodeInfoSnapshot: snapshot,
			}

			podGroupInfo := &framework.QueuedPodGroupInfo{
				PodGroupInfo: &framework.PodGroupInfo{
					Name:      tt.podGroup.Name,
					Namespace: tt.podGroup.Namespace,
					Type:      fwk.PodGroupKeyType,
					PodGroup:  tt.podGroup,
				},
				QueuedPodInfos: make(map[fwk.EntityKey][]*framework.QueuedPodInfo),
			}
			for _, pod := range tt.pods {
				podGroupInfo.UnscheduledPods = append(podGroupInfo.UnscheduledPods, pod)
				key := fwk.PodGroupKey(tt.podGroup.Namespace, tt.podGroup.Name)
				podGroupInfo.QueuedPodInfos[key] = append(podGroupInfo.QueuedPodInfos[key],
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

	testPodGroup := &schedulingv1beta1.PodGroup{
		ObjectMeta: metav1.ObjectMeta{Name: "pg", Namespace: "default"},
	}

	podGroupInfo := &framework.QueuedPodGroupInfo{
		QueuedPodInfos: map[fwk.EntityKey][]*framework.QueuedPodInfo{fwk.MustParseEntityKey("podgroup/default/pg"): {qInfo1, qInfo2, qInfo3}},
		PodGroupInfo: &framework.PodGroupInfo{
			Name:            "pg",
			Namespace:       "default",
			Type:            fwk.PodGroupKeyType,
			UnscheduledPods: []*v1.Pod{p1, p2, p3},
			PodGroup:        testPodGroup,
		},
	}

	logger, ctx := ktesting.NewTestContext(t)

	cache := internalcache.New(ctx, nil, true, true /* CompositePodGroup */)
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

	if podGroupInfo.Size() != 1 {
		t.Errorf("Expected 1 queued pod left, got %d", podGroupInfo.Size())
	}
	if podGroupInfo.QueuedPodInfos[fwk.MustParseEntityKey("podgroup/default/pg")][0].Pod.Name != "p1" {
		t.Errorf("Expected p1 to be left in queued pods, got %s", podGroupInfo.QueuedPodInfos[fwk.MustParseEntityKey("podgroup/default/pg")][0].Pod.Name)
	}
	if len(podGroupInfo.UnscheduledPods) != 1 {
		t.Errorf("Expected 1 unscheduled pod left, got %d", len(podGroupInfo.UnscheduledPods))
	}
	if podGroupInfo.UnscheduledPods[0].Name != "p1" {
		t.Errorf("Expected p1 to be left in unscheduled pods, got %s", podGroupInfo.UnscheduledPods[0].Name)
	}
}

func TestScheduleOnePodGroup_FinishesAttemptWhenAllPoppedPodsAreAssumed(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, true)

	tests := []struct {
		name                       string
		memberArrivesWhileInFlight bool
	}{
		{
			name:                       "pending member is requeued",
			memberArrivesWhileInFlight: true,
		},
		{
			name:                       "later member starts a new queued PodGroup",
			memberArrivesWhileInFlight: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			podGroup := st.MakePodGroup().Name("pg").Namespace("default").Obj()
			p1 := st.MakePod().Name("p1").Namespace("default").UID("p1").PodGroupName(podGroup.Name).SchedulerName("test-scheduler").Obj()
			p2 := st.MakePod().Name("p2").Namespace("default").UID("p2").PodGroupName(podGroup.Name).SchedulerName("test-scheduler").Obj()

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

			cache := internalcache.New(ctx, nil, true, false)
			cache.AddPodGroup(podGroup)
			cache.AddPodGroupMember(p1)
			assumedP1 := p1.DeepCopy()
			assumedP1.Spec.NodeName = "node1"
			if err := cache.AssumePod(logger, assumedP1); err != nil {
				t.Fatalf("Failed to assume pod: %v", err)
			}

			queue := internalqueue.NewTestQueue(ctx, schedFwk.QueueSortFunc())
			queue.AddPodGroup(logger, podGroup)
			queue.Add(ctx, p1)
			entity, err := queue.Pop(logger)
			if err != nil {
				t.Fatalf("Failed to pop pod group: %v", err)
			}
			podGroupInfo := entity.(*framework.QueuedPodGroupInfo)

			if tt.memberArrivesWhileInFlight {
				// A member arriving while its PodGroup is scheduling waits for
				// that scheduling attempt to finish.
				queue.Add(ctx, p2)
				if pendingPods := queue.PendingPodGroupPods(); len(pendingPods) != 1 || pendingPods[0].UID != p2.UID {
					t.Fatalf("Expected pod %q to be pending, got %v", p2.Name, pendingPods)
				}
			}

			sched := &Scheduler{
				Profiles:         profile.Map{"test-scheduler": schedFwk},
				Cache:            cache,
				nodeInfoSnapshot: internalcache.NewEmptySnapshot(),
				SchedulingQueue:  queue,
			}
			sched.scheduleOnePodGroup(ctx, podGroupInfo)

			if !tt.memberArrivesWhileInFlight {
				queue.Add(ctx, p2)
			}
			if pendingPods := queue.PendingPodGroupPods(); len(pendingPods) != 0 {
				t.Errorf("Expected no pending PodGroup members, got %v", pendingPods)
			}
			requeuedPodGroup, ok := queue.GetPodGroup(podGroup.Name, podGroup.Namespace, fwk.PodGroupKeyType)
			if !ok {
				t.Fatalf("Expected PodGroup to be queued")
			}
			if len(requeuedPodGroup.QueuedPodInfos) != 1 {
				t.Errorf("Expected 1 key in QueuedPodInfos, got %v", requeuedPodGroup.QueuedPodInfos)
			}
			infos := requeuedPodGroup.QueuedPodInfos[fwk.MustParseEntityKey("podgroup/default/pg")]
			if len(infos) != 1 || infos[0].Pod.UID != p2.UID {
				t.Errorf("Expected queued PodGroup to contain pod %q, got %v", p2.Name, infos)
			}
		})
	}
}

func TestPodGroupCycle_UpdateSnapshotError(t *testing.T) {
	p1 := st.MakePod().Name("p1").UID("p1").PodGroupName("pg").SchedulerName("test-scheduler").Obj()
	p2 := st.MakePod().Name("p2").UID("p2").PodGroupName("pg").SchedulerName("test-scheduler").Obj()
	qInfo1 := &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: p1}}
	qInfo2 := &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: p2}}

	testPodGroup := &schedulingv1beta1.PodGroup{
		ObjectMeta: metav1.ObjectMeta{Name: "pg", Namespace: "default"},
	}

	podGroupInfo := &framework.QueuedPodGroupInfo{
		QueuedPodInfos: map[fwk.EntityKey][]*framework.QueuedPodInfo{fwk.MustParseEntityKey("podgroup/default/pg"): {qInfo1, qInfo2}},
		PodGroupInfo: &framework.PodGroupInfo{
			Name:            "pg",
			Namespace:       "default",
			Type:            fwk.PodGroupKeyType,
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
		Cache: internalcache.New(ctx, nil, true, false /* CompositePodGroup */),
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
	queuedPodInfos := []*framework.QueuedPodInfo{qInfo1, qInfo2, qInfo3}

	podGroupInfo := &framework.QueuedPodGroupInfo{
		QueuedPodInfos: map[fwk.EntityKey][]*framework.QueuedPodInfo{fwk.MustParseEntityKey("podgroup/default/pg"): queuedPodInfos},
		PodGroupInfo: &framework.PodGroupInfo{
			Name:            "pg",
			Namespace:       "default",
			Type:            fwk.PodGroupKeyType,
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

	cache := internalcache.New(ctx, nil, true, true /* CompositePodGroup */)
	logger, ctx := ktesting.NewTestContext(t)
	cache.AddNode(logger, testNode)
	cache.AddPodGroup(testPodGroup)

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

	resultsMap := sched.runRootSchedulingAlgorithm(ctx, schedFwk, framework.NewCycleState(), podGroupInfo)
	schedulePodResult := resultsMap[pgKey(podGroupInfo.PodGroupInfo)]
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
				QueuedPodInfos: map[fwk.EntityKey][]*framework.QueuedPodInfo{fwk.MustParseEntityKey("podgroup/default/pg"): {qInfo1, qInfo2}},
				PodGroupInfo: &framework.PodGroupInfo{
					Name:            "pg",
					Namespace:       "default",
					Type:            fwk.PodGroupKeyType,
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

			cache := internalcache.New(ctx, nil, true, true /* CompositePodGroup */)
			logger, ctx := ktesting.NewTestContext(t)
			cache.AddNode(logger, testNode)
			cache.AddPodGroup(testPodGroup)

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
			if err := sched.Cache.UpdateSnapshot(logger, sched.nodeInfoSnapshot); err != nil {
				t.Fatalf("Failed to update snapshot: %v", err)
			}
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

	testPodGroup := &schedulingv1beta1.PodGroup{
		ObjectMeta: metav1.ObjectMeta{Name: "pg", Namespace: "default"},
	}

	qInfo1 := &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: p1}}
	qInfo2 := &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: p2}}
	qInfo3 := &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: p3}}
	queuedPodInfos := []*framework.QueuedPodInfo{qInfo1, qInfo2, qInfo3}

	podGroupInfo := &framework.QueuedPodGroupInfo{
		QueuedPodInfos: map[fwk.EntityKey][]*framework.QueuedPodInfo{fwk.MustParseEntityKey("podgroup/default/pg"): queuedPodInfos},
		PodGroupInfo: &framework.PodGroupInfo{
			Name:            "pg",
			Namespace:       "default",
			Type:            fwk.PodGroupKeyType,
			UnscheduledPods: []*v1.Pod{p1, p2, p3},
			PodGroup:        testPodGroup,
		},
	}

	tests := []struct {
		name                     string
		plugin                   *fakePodGroupPlugin
		podGroupFeasibleStatuses []fwk.Code
		expectedGroupStatusCode  fwk.Code
		expectedPodStatus        map[string]*fwk.Status
		skipForTAS               bool
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
			name: "PodGroup schedulable with 2 schedulable pods",
			plugin: &fakePodGroupPlugin{
				filterStatus: map[string]*fwk.Status{
					"p1": fwk.NewStatus(fwk.Unschedulable),
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
				"p1": fwk.NewStatus(fwk.Unschedulable),
				"p2": nil,
				"p3": nil,
			},
			skipForTAS: false,
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
	}

	for _, tasEnabled := range []bool{true, false} {
		for _, cpgEnabled := range []bool{false, true} {
			if !tasEnabled && cpgEnabled {
				continue
			}
			for _, tt := range tests {
				if tasEnabled && tt.skipForTAS {
					continue
				}
				name := fmt.Sprintf("%s (TopologyAwareWorkloadScheduling=%v, CompositePodGroup=%v)", tt.name, tasEnabled, cpgEnabled)
				t.Run(name, func(t *testing.T) {
					featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
						features.TopologyAwareWorkloadScheduling: tasEnabled,
						features.GenericWorkload:                 true,
						features.CompositePodGroup:               cpgEnabled,
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

					cache := internalcache.New(ctx, nil, true, cpgEnabled /* CompositePodGroup */)
					cache.AddNode(logger, testNode)
					cache.AddPodGroup(testPodGroup)

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

					resultsMap := sched.runRootSchedulingAlgorithm(ctx, schedFwk, framework.NewCycleState(), podGroupInfo)
					result := resultsMap[pgKey(podGroupInfo.PodGroupInfo)]

					if result.status.Code() != tt.expectedGroupStatusCode {
						t.Errorf("Expected group status code: %v, got: %v", tt.expectedGroupStatusCode, result.status.Code())
					}
					if len(tt.expectedPodStatus) != len(result.podResults) {
						t.Errorf("Expected %d pod results, got %d", len(tt.expectedPodStatus), len(result.podResults))
					}
					for _, podResult := range result.podResults {
						podName := podResult.podInfo.Pod.Name
						if expected, ok := tt.expectedPodStatus[podName]; ok {
							if podResult.status.Code() != expected.Code() {
								t.Errorf("Expected pod %s status code: %v, got: %v", podName, expected.Code(), podResult.status.Code())
							}
						} else {
							t.Errorf("Got result for unexpected pod %s: %v", podName, podResult.status.Code())
						}
						if podResult.status.IsSuccess() {
							if podResult.scheduleResult.SuggestedHost != "node1" {
								t.Errorf("Expected pod %s suggested host: node1, got: %v", podName, podResult.scheduleResult.SuggestedHost)
							}
						} else {
							if podResult.scheduleResult.SuggestedHost != "" {
								t.Errorf("Expected pod %s empty suggested host, got: %v", podName, podResult.scheduleResult.SuggestedHost)
							}
						}
					}
				})
			}
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

	testPodGroup := &schedulingv1beta1.PodGroup{
		ObjectMeta: metav1.ObjectMeta{Name: "pg", Namespace: "default"},
	}

	tests := []struct {
		name             string
		existingPodGroup *schedulingv1beta1.PodGroup
		// podResultsByPodName maps each pod name to its scheduling algorithm result.
		// The final podResults slice is built in the order of UnscheduledPods after Pop,
		// making the test independent of pod ordering within the pod group.
		podResultsByPodName     map[string]algorithmResult
		status                  *fwk.Status
		waitingOnPreemption     bool
		expectBound             sets.Set[string]
		expectPreempting        map[string]string
		expectFailed            sets.Set[string]
		expectCondition         *metav1.Condition
		expectPodsInActiveQueue sets.Set[string]
	}{
		{
			name: "All pods feasible",
			podResultsByPodName: map[string]algorithmResult{
				"p1": {scheduleResult: ScheduleResult{SuggestedHost: "node1"}, status: nil},
				"p2": {scheduleResult: ScheduleResult{SuggestedHost: "node1"}, status: nil},
				"p3": {scheduleResult: ScheduleResult{SuggestedHost: "node1"}, status: nil},
			},
			expectBound: sets.New("p1", "p2", "p3"),
			expectCondition: &metav1.Condition{
				Type:   schedulingapi.PodGroupInitiallyScheduled,
				Status: metav1.ConditionTrue,
				Reason: "Scheduled",
			},
		},
		{
			name:   "All pods feasible, but podGroup unschedulable",
			status: fwk.NewStatus(fwk.Unschedulable, "not enough capacity for the gang"),
			podResultsByPodName: map[string]algorithmResult{
				"p1": {scheduleResult: ScheduleResult{SuggestedHost: "node1"}, status: nil},
				"p2": {scheduleResult: ScheduleResult{SuggestedHost: "node1"}, status: nil},
				"p3": {scheduleResult: ScheduleResult{SuggestedHost: "node1"}, status: nil},
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
			podResultsByPodName: map[string]algorithmResult{
				"p1": {scheduleResult: ScheduleResult{SuggestedHost: "node1"}, status: nil},
				"p2": {scheduleResult: ScheduleResult{SuggestedHost: "", nominatingInfo: clearNominatedNode}, status: fwk.NewStatus(fwk.Unschedulable)},
				"p3": {scheduleResult: ScheduleResult{SuggestedHost: "node1"}, status: nil},
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
			name:   "All pods unschedulable",
			status: fwk.NewStatus(fwk.Unschedulable, "0/3 nodes are available: insufficient cpu"),
			podResultsByPodName: map[string]algorithmResult{
				"p1": {scheduleResult: ScheduleResult{SuggestedHost: "", nominatingInfo: clearNominatedNode}, status: fwk.NewStatus(fwk.Unschedulable)},
				"p2": {scheduleResult: ScheduleResult{SuggestedHost: "", nominatingInfo: clearNominatedNode}, status: fwk.NewStatus(fwk.Unschedulable)},
				"p3": {scheduleResult: ScheduleResult{SuggestedHost: "", nominatingInfo: clearNominatedNode}, status: fwk.NewStatus(fwk.Unschedulable)},
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
			name:   "All pods unschedulable with nil nominatingInfo",
			status: fwk.NewStatus(fwk.Unschedulable, "0/3 nodes are available: insufficient cpu"),
			podResultsByPodName: map[string]algorithmResult{
				"p1": {scheduleResult: ScheduleResult{SuggestedHost: "", nominatingInfo: nil}, status: fwk.NewStatus(fwk.Unschedulable)},
				"p2": {scheduleResult: ScheduleResult{SuggestedHost: "", nominatingInfo: nil}, status: fwk.NewStatus(fwk.Unschedulable)},
				"p3": {scheduleResult: ScheduleResult{SuggestedHost: "", nominatingInfo: nil}, status: fwk.NewStatus(fwk.Unschedulable)},
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
			name:   "Unschedulable for the entire pod group",
			status: fwk.NewStatus(fwk.Unschedulable, "node affinity mismatch"),
			podResultsByPodName: map[string]algorithmResult{
				"p1": {scheduleResult: ScheduleResult{SuggestedHost: "", nominatingInfo: clearNominatedNode}, status: fwk.NewStatus(fwk.Unschedulable)},
				"p2": {scheduleResult: ScheduleResult{SuggestedHost: "", nominatingInfo: clearNominatedNode}, status: fwk.NewStatus(fwk.Unschedulable)},
				"p3": {scheduleResult: ScheduleResult{SuggestedHost: "", nominatingInfo: clearNominatedNode}, status: fwk.NewStatus(fwk.Unschedulable)},
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
			name:   "Error for one pod",
			status: fwk.NewStatus(fwk.Error, "plugin returned error"),
			podResultsByPodName: map[string]algorithmResult{
				"p1": {scheduleResult: ScheduleResult{SuggestedHost: "node1"}, status: nil},
				"p2": {scheduleResult: ScheduleResult{SuggestedHost: "", nominatingInfo: clearNominatedNode}, status: fwk.NewStatus(fwk.Error, "plugin returned error")},
				"p3": {scheduleResult: ScheduleResult{SuggestedHost: "", nominatingInfo: clearNominatedNode}, status: fwk.NewStatus(fwk.Error, "plugin returned error")},
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
			name: "Already Scheduled, successful cycle keeps condition",
			existingPodGroup: &schedulingv1beta1.PodGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "pg", Namespace: "default"},
				Status: schedulingv1beta1.PodGroupStatus{
					Conditions: []metav1.Condition{{
						Type:               schedulingapi.PodGroupInitiallyScheduled,
						Status:             metav1.ConditionTrue,
						Reason:             "Scheduled",
						Message:            "All pods scheduled",
						LastTransitionTime: metav1.Now(),
					}},
				},
			},
			podResultsByPodName: map[string]algorithmResult{
				"p1": {scheduleResult: ScheduleResult{SuggestedHost: "node1"}, status: nil},
				"p2": {scheduleResult: ScheduleResult{SuggestedHost: "node1"}, status: nil},
				"p3": {scheduleResult: ScheduleResult{SuggestedHost: "node1"}, status: nil},
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
			existingPodGroup: &schedulingv1beta1.PodGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "pg", Namespace: "default"},
				Status: schedulingv1beta1.PodGroupStatus{
					Conditions: []metav1.Condition{{
						Type:               schedulingapi.PodGroupInitiallyScheduled,
						Status:             metav1.ConditionTrue,
						Reason:             "Scheduled",
						Message:            "All pods scheduled",
						LastTransitionTime: metav1.Now(),
					}},
				},
			},
			status: fwk.NewStatus(fwk.Unschedulable, "extra pods could not be placed"),
			podResultsByPodName: map[string]algorithmResult{
				"p1": {scheduleResult: ScheduleResult{SuggestedHost: "", nominatingInfo: clearNominatedNode}, status: fwk.NewStatus(fwk.Unschedulable)},
				"p2": {scheduleResult: ScheduleResult{SuggestedHost: "", nominatingInfo: clearNominatedNode}, status: fwk.NewStatus(fwk.Unschedulable)},
				"p3": {scheduleResult: ScheduleResult{SuggestedHost: "", nominatingInfo: clearNominatedNode}, status: fwk.NewStatus(fwk.Unschedulable)},
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
			existingPodGroup: &schedulingv1beta1.PodGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "pg", Namespace: "default"},
				Status: schedulingv1beta1.PodGroupStatus{
					Conditions: []metav1.Condition{{
						Type:               schedulingapi.PodGroupInitiallyScheduled,
						Status:             metav1.ConditionTrue,
						Reason:             "Scheduled",
						Message:            "All pods scheduled",
						LastTransitionTime: metav1.Now(),
					}},
				},
			},
			status: fwk.NewStatus(fwk.Error),
			podResultsByPodName: map[string]algorithmResult{
				"p1": {scheduleResult: ScheduleResult{SuggestedHost: "node1"}, status: nil},
				"p2": {scheduleResult: ScheduleResult{SuggestedHost: "", nominatingInfo: clearNominatedNode}, status: fwk.NewStatus(fwk.Error)},
				"p3": {scheduleResult: ScheduleResult{SuggestedHost: "", nominatingInfo: clearNominatedNode}, status: fwk.NewStatus(fwk.Error)},
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
			// If:
			// - PodGroup failed scheduling
			// - some pods were successfully evaluated
			// - pod group preemption successfully found placement
			// The scheduler should use the nominating info from preemption instead of suggested host from evaluation.
			name:                "PodGroup waiting on preemption uses nominating info instead of suggested host for successfully evaluated pods",
			status:              fwk.NewStatus(fwk.Unschedulable, "waiting on preemption"),
			waitingOnPreemption: true,
			podResultsByPodName: map[string]algorithmResult{
				"p1": {scheduleResult: ScheduleResult{SuggestedHost: "node1", nominatingInfo: &fwk.NominatingInfo{NominatedNodeName: "node2", NominatingMode: fwk.ModeOverride}}, status: nil},
				"p2": {scheduleResult: ScheduleResult{SuggestedHost: "", nominatingInfo: clearNominatedNode}, status: fwk.NewStatus(fwk.Unschedulable)},
				"p3": {scheduleResult: ScheduleResult{SuggestedHost: "", nominatingInfo: clearNominatedNode}, status: fwk.NewStatus(fwk.Unschedulable)},
			},
			expectPreempting: map[string]string{"p1": "node2"},
			expectFailed:     sets.New("p2", "p3"),
			expectCondition: &metav1.Condition{
				Type:    schedulingapi.PodGroupInitiallyScheduled,
				Status:  metav1.ConditionFalse,
				Reason:  schedulingapi.PodGroupReasonUnschedulable,
				Message: "waiting on preemption",
			},
		},
		{
			name:   "Different number of pods in result and queue, should fail all queue pods",
			status: fwk.NewStatus(fwk.Error),
			podResultsByPodName: map[string]algorithmResult{
				"p1": {scheduleResult: ScheduleResult{SuggestedHost: "node1"}, status: nil},
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
			preemptingPods := make(map[string]string)
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

			cache := internalcache.New(ctx, nil, true, true /* CompositePodGroup */)
			cache.AddNode(klog.FromContext(ctx), testNode)

			informerFactory := informers.NewSharedInformerFactory(client, 0)
			informerFactory.Start(ctx.Done())
			informerFactory.WaitForCacheSync(ctx.Done())

			fakeClock := testingclock.NewFakeClock(time.Now())
			schedulingQueue := internalqueue.NewTestQueue(ctx, schedFwk.QueueSortFunc(), internalqueue.WithClock(fakeClock))
			sched := &Scheduler{
				client:          client,
				Cache:           cache,
				Profiles:        profile.Map{"test-scheduler": schedFwk},
				SchedulingQueue: schedulingQueue,
				FailureHandler: func(ctx context.Context, fwk framework.Framework, p *framework.QueuedPodInfo, status *fwk.Status, ni *fwk.NominatingInfo, start time.Time) {
					lock.Lock()
					if ni != nil && ni.NominatedNodeName != "" {
						preemptingPods[p.Pod.Name] = ni.NominatedNodeName
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
			// Advance the clock between additions to keep pod ordering deterministic.
			schedulingQueue.Add(ctx, p1)
			fakeClock.Step(time.Second)
			schedulingQueue.Add(ctx, p2)
			fakeClock.Step(time.Second)
			schedulingQueue.Add(ctx, p3)
			entity, err := schedulingQueue.Pop(logger)
			if err != nil {
				t.Fatalf("Failed to pop pod group: %v", err)
			}
			podGroupInfo := entity.(*framework.QueuedPodGroupInfo)
			podGroupInfo.PodGroup = pg
			oldTimestamp := podGroupInfo.Timestamp

			podGroupCycleState := framework.NewCycleState()

			// Build the podResults slice in the order of UnscheduledPods after Pop,
			// so the test is independent of pod ordering within the pod group.
			algorithmResult := podGroupAlgorithmResult{
				podGroupInfo:        podGroupInfo.PodGroupInfo,
				status:              tt.status,
				waitingOnPreemption: tt.waitingOnPreemption,
			}
			for _, pod := range podGroupInfo.UnscheduledPods {
				result, ok := tt.podResultsByPodName[pod.Name]
				if !ok {
					// This pod was not processed by the algorithm (fewer results than pods in queue).
					// Skip it — the test verifies that all queue pods are failed in this scenario.
					continue
				}
				placementCycleState := framework.NewCycleState()
				placementCycleState.SetPodGroupSchedulingCycle(podGroupCycleState)
				result.podCtx = initPodSchedulingContext(ctx, pod, placementCycleState)
				algorithmResult.podResults = append(algorithmResult.podResults, result)
			}

			sched.submitPodGroupAlgorithmResult(ctx, schedFwk, podGroupCycleState, podGroupInfo, map[fwk.EntityKey]*podGroupAlgorithmResult{pgKey(algorithmResult.podGroupInfo): &algorithmResult}, time.Now(), tt.status)

			if err := wait.PollUntilContextTimeout(ctx, time.Millisecond*200, wait.ForeverTestTimeout, false, func(ctx context.Context) (bool, error) {
				lock.Lock()
				defer lock.Unlock()
				return len(boundPods)+len(preemptingPods)+len(failedPods) == podGroupInfo.Size(), nil
			}); err != nil {
				t.Errorf("Failed waiting for all pods to be either bound or failed")
			}

			if !tt.expectBound.Equal(boundPods) {
				t.Errorf("Expected bound pods: %v, but got: %v", tt.expectBound, boundPods)
			}
			if diff := cmp.Diff(tt.expectPreempting, preemptingPods, cmpopts.EquateEmpty()); diff != "" {
				t.Errorf("Unexpected preempting pods (-want, +got):\n%s", diff)
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
				queuedPGInfo, ok := sched.SchedulingQueue.GetPodGroup("pg", "default", fwk.PodGroupKeyType)
				if !ok {
					t.Errorf("Expected pod group pg to be requeued, but it was not found in the scheduling queue")
				} else if !queuedPGInfo.Timestamp.Equal(oldTimestamp) {
					t.Errorf("Expected timestamp to be preserved exactly for pod group. Original: %v, Requeued: %v", oldTimestamp, queuedPGInfo.Timestamp)
				}
			}

			updatedPodGroup, err := client.SchedulingV1beta1().PodGroups("default").Get(ctx, "pg", metav1.GetOptions{})
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
		existingPodGroup *schedulingv1beta1.PodGroup
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
			existingPodGroup: &schedulingv1beta1.PodGroup{
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
			existingPodGroup: &schedulingv1beta1.PodGroup{
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
			existingPodGroup: &schedulingv1beta1.PodGroup{
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
			existingPodGroup: &schedulingv1beta1.PodGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "pg4", Namespace: "ns1"},
				Status: schedulingv1beta1.PodGroupStatus{
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
			existingPodGroup: &schedulingv1beta1.PodGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "pg-se-to-true", Namespace: "ns1"},
				Status: schedulingv1beta1.PodGroupStatus{
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
			existingPodGroup: &schedulingv1beta1.PodGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "pg-true-to-unsched", Namespace: "ns1"},
				Status: schedulingv1beta1.PodGroupStatus{
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
			existingPodGroup: &schedulingv1beta1.PodGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "pg-true-to-se", Namespace: "ns1"},
				Status: schedulingv1beta1.PodGroupStatus{
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
			existingPodGroup: &schedulingv1beta1.PodGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "pg-unsched-to-se", Namespace: "ns1"},
				Status: schedulingv1beta1.PodGroupStatus{
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
			existingPodGroup: &schedulingv1beta1.PodGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "pg-se-to-unsched", Namespace: "ns1"},
				Status: schedulingv1beta1.PodGroupStatus{
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
			existingPodGroup: &schedulingv1beta1.PodGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "pg-true-to-true", Namespace: "ns1"},
				Status: schedulingv1beta1.PodGroupStatus{
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
			existingPodGroup: &schedulingv1beta1.PodGroup{
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
					Type:      fwk.PodGroupKeyType,
				},
			}
			sched.updatePodGroupCondition(ctx, podGroupInfo.PodGroupInfo, tt.condition)

			updatedPodGroup, err := client.SchedulingV1beta1().PodGroups(tt.namespace).Get(ctx, tt.podGroupName, metav1.GetOptions{})
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
	filterStatus             map[string]*fwk.Status            // node name to status
	generatePlacementsResult map[string]map[string][]string    // PodGroupInfo key to placement name to list of node names
	generatePlacementsStatus map[string]*fwk.Status            // PodGroupInfo key to status
	scorePlacementsResult    map[string]map[string]int64       // PodGroupInfo key to placement name to score
	scorePlacementsStatus    map[string]map[string]*fwk.Status // PodGroupInfo key to placement name to status
	podPerNode               bool
	reservedNodes            sets.Set[string]
}

var _ fwk.FilterPlugin = &fakePlacementPlugin{}
var _ fwk.PlacementGeneratePlugin = &fakePlacementPlugin{}
var _ fwk.PlacementScorePlugin = &fakePlacementPlugin{}
var _ fwk.ReservePlugin = &fakePlacementPlugin{}

func (mp *fakePlacementPlugin) Name() string { return mp.name }

func (mp *fakePlacementPlugin) Reserve(ctx context.Context, state fwk.CycleState, p *v1.Pod, nodeName string) *fwk.Status {
	if mp.podPerNode {
		mp.reservedNodes.Insert(nodeName)
	}
	return nil
}

func (mp *fakePlacementPlugin) Unreserve(ctx context.Context, state fwk.CycleState, p *v1.Pod, nodeName string) {
	if mp.podPerNode {
		mp.reservedNodes.Delete(nodeName)
	}
}

func (mp *fakePlacementPlugin) Filter(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeInfo fwk.NodeInfo) *fwk.Status {
	if status := mp.filterStatus[nodeInfo.Node().Name]; status != nil {
		return status
	}
	if mp.reservedNodes.Has(nodeInfo.Node().Name) {
		return fwk.NewStatus(fwk.Unschedulable, fmt.Sprintf("%s is already reserved", nodeInfo.Node().Name))
	}
	return nil
}

func (mp *fakePlacementPlugin) PlacementScoreExtensions() fwk.PlacementScoreExtensions {
	return nil
}

func (mp *fakePlacementPlugin) ScorePlacement(ctx context.Context, state fwk.PlacementCycleState, podGroup fwk.PodGroupInfo, placement *fwk.PodGroupAssignments) (int64, *fwk.Status) {
	return mp.scorePlacementsResult[podGroup.GetKey()][placement.Name], mp.scorePlacementsStatus[podGroup.GetKey()][placement.Name]
}

func (mp *fakePlacementPlugin) GeneratePlacements(ctx context.Context, state fwk.PodGroupCycleState, podGroup fwk.PodGroupInfo, parentPlacement *fwk.Placement) (*fwk.GeneratePlacementsResult, *fwk.Status) {
	parentNodes := map[string]fwk.NodeInfo{}
	for _, node := range parentPlacement.Nodes {
		parentNodes[node.Node().Name] = node
	}

	generatePlacementsResult := mp.generatePlacementsResult[podGroup.GetKey()]
	placements := make([]*fwk.Placement, 0, len(generatePlacementsResult))
	for placementName, nodeNames := range generatePlacementsResult {
		placement := &fwk.Placement{Name: placementName}
		for _, nodeName := range nodeNames {
			if node, ok := parentNodes[nodeName]; ok && node != nil {
				placement.Nodes = append(placement.Nodes, node)
			}
		}
		if len(placement.Nodes) > 0 {
			placements = append(placements, placement)
		}
	}
	return &fwk.GeneratePlacementsResult{Placements: placements}, mp.generatePlacementsStatus[podGroup.GetKey()]
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

func assertCounterValueFromGatherer(t *testing.T, g componentmetrics.Gatherer, name, labelName, labelValue string, want int) {
	t.Helper()
	got := 0
	if vals, err := testutil.GetCounterValuesFromGatherer(g, name, nil, labelName); err == nil {
		got = int(vals[labelValue])
	}
	if got != want {
		t.Errorf("unexpected %s{%s=%q}: got %d, want %d", name, labelName, labelValue, got, want)
	}
}

func assertHistogramSampleCountFromGatherer(t *testing.T, g componentmetrics.Gatherer, name string, labels map[string]string, want int) {
	t.Helper()
	got := 0
	if vec, err := testutil.GetHistogramVecFromGatherer(g, name, labels); err == nil {
		got = int(vec.GetAggregatedSampleCount())
	}
	if got != want {
		t.Errorf("unexpected %s%v sample count: got %d, want %d", name, labels, got, want)
	}
}

func TestPodGroupSchedulingPlacementAlgorithm(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.TopologyAwareWorkloadScheduling: true,
		features.GenericWorkload:                 true,
	})
	testRegistry := componentmetrics.NewKubeRegistry()
	testRegistry.MustRegister(metrics.GeneratedPlacementsTotal, metrics.PlacementEvaluations, metrics.PlacementEvaluationDuration)

	nodes := []*v1.Node{
		st.MakeNode().Name("node1").Obj(),
		st.MakeNode().Name("node2").Obj(),
	}
	podGroupPod := st.MakePod().Name("foo").UID("foo").PodGroupName("pg").Obj()
	testPodGroup := &schedulingv1beta1.PodGroup{
		ObjectMeta: metav1.ObjectMeta{Name: "pg", Namespace: "default"},
	}

	podInfo, err := framework.NewPodInfo(st.MakePod().Name("foo").UID("foo").PodGroupName("pg").Obj())
	if err != nil {
		t.Fatalf("Failed to create pod info: %v", err)
	}
	podGroupPodInfo := &framework.QueuedPodInfo{PodInfo: podInfo}

	queuedPodInfos := []*framework.QueuedPodInfo{{PodInfo: &framework.PodInfo{Pod: podGroupPod}}}
	pgInfo := &framework.QueuedPodGroupInfo{
		QueuedPodInfos: map[fwk.EntityKey][]*framework.QueuedPodInfo{fwk.MustParseEntityKey("podgroup/default/pg"): queuedPodInfos},
		PodGroupInfo: &framework.PodGroupInfo{
			Name:            "pg",
			Namespace:       "default",
			Type:            fwk.PodGroupKeyType,
			PodGroup:        testPodGroup,
			UnscheduledPods: []*v1.Pod{podGroupPod},
		},
	}

	tests := map[string]struct {
		placementPlugin               fakePlacementPlugin
		placementFeasibleStatuses     [][]fwk.Code
		expectedResult                podGroupAlgorithmResult
		expectedGeneratedPlacements   int
		expectedFeasibleEvaluations   int
		expectedInfeasibleEvaluations int
	}{
		"respects higher score of placement1": {
			expectedGeneratedPlacements: 2,
			expectedFeasibleEvaluations: 2,
			placementPlugin: fakePlacementPlugin{
				generatePlacementsResult: map[string]map[string][]string{
					pgInfo.GetKey(): {
						"placement1": {nodes[0].Name},
						"placement2": {nodes[1].Name},
					},
				},
				scorePlacementsResult: map[string]map[string]int64{
					pgInfo.GetKey(): {
						"placement1": 2,
						"placement2": 1,
					},
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
			expectedGeneratedPlacements: 2,
			expectedFeasibleEvaluations: 2,
			placementPlugin: fakePlacementPlugin{
				generatePlacementsResult: map[string]map[string][]string{
					pgInfo.GetKey(): {
						"placement1": {nodes[0].Name},
						"placement2": {nodes[1].Name},
					},
				},
				scorePlacementsResult: map[string]map[string]int64{
					pgInfo.GetKey(): {
						"placement1": 1,
						"placement2": 2,
					},
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
				generatePlacementsResult: map[string]map[string][]string{},
			},
			expectedResult: podGroupAlgorithmResult{
				status: fwk.NewStatus(fwk.Unschedulable, "no feasible placements found").WithPlugin("FakePlacementPlugin_Ordered"),
			},
		},
		"when all placements are infeasible, returns unschedulable": {
			expectedGeneratedPlacements:   2,
			expectedInfeasibleEvaluations: 2,
			placementPlugin: fakePlacementPlugin{
				generatePlacementsResult: map[string]map[string][]string{
					pgInfo.GetKey(): {
						"placement1": {nodes[0].Name},
						"placement2": {nodes[1].Name},
					},
				},
				scorePlacementsResult: map[string]map[string]int64{
					pgInfo.GetKey(): {
						"placement1": 1,
						"placement2": 2,
					},
				},
				filterStatus: map[string]*fwk.Status{
					nodes[0].Name: fwk.NewStatus(fwk.Unschedulable),
					nodes[1].Name: fwk.NewStatus(fwk.Unschedulable),
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
							EvaluatedNodes: 0,
							FeasibleNodes:  0,
						},
						status: fwk.NewStatus(fwk.Unschedulable, "0/1 nodes are available:"),
					},
				},
				status: fwk.NewStatus(fwk.Unschedulable, "0/2 placements are available, first placement status: injected placementFeasible status"),
			},
		},
		"when all placements are infeasible, but pods are feasible, returns unschedulable": {
			expectedGeneratedPlacements:   2,
			expectedInfeasibleEvaluations: 2,
			placementPlugin: fakePlacementPlugin{
				generatePlacementsResult: map[string]map[string][]string{
					pgInfo.GetKey(): {
						"placement1": {nodes[0].Name},
						"placement2": {nodes[1].Name},
					},
				},
				scorePlacementsResult: map[string]map[string]int64{
					pgInfo.GetKey(): {
						"placement1": 1,
						"placement2": 2,
					},
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
			expectedGeneratedPlacements:   2,
			expectedFeasibleEvaluations:   1,
			expectedInfeasibleEvaluations: 1,
			placementPlugin: fakePlacementPlugin{
				generatePlacementsResult: map[string]map[string][]string{
					pgInfo.GetKey(): {
						"placement1": {nodes[0].Name},
						"placement2": {nodes[1].Name},
					},
				},
				scorePlacementsResult: map[string]map[string]int64{
					pgInfo.GetKey(): {
						"placement1": 1,
					},
				},
				filterStatus: map[string]*fwk.Status{
					nodes[1].Name: fwk.NewStatus(fwk.Unschedulable),
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
		"filters out infeasible placements with feasible pods": {
			expectedGeneratedPlacements:   2,
			expectedFeasibleEvaluations:   1,
			expectedInfeasibleEvaluations: 1,
			placementPlugin: fakePlacementPlugin{
				generatePlacementsResult: map[string]map[string][]string{
					pgInfo.GetKey(): {
						"placement1": {nodes[0].Name},
						"placement2": {nodes[1].Name},
					},
				},
				scorePlacementsResult: map[string]map[string]int64{
					pgInfo.GetKey(): {
						"placement1": 1,
						"placement2": 2,
					},
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
				generatePlacementsStatus: map[string]*fwk.Status{pgInfo.GetKey(): fwk.NewStatus(fwk.Error, "error for test")},
			},
			expectedResult: podGroupAlgorithmResult{
				status: fwk.NewStatus(fwk.Error, "error for test").WithPlugin("FakePlacementPlugin"),
			},
		},
		"when score plugin fails, returns error": {
			expectedGeneratedPlacements: 2,
			expectedFeasibleEvaluations: 2,
			placementPlugin: fakePlacementPlugin{
				generatePlacementsResult: map[string]map[string][]string{
					pgInfo.GetKey(): {
						"placement1": {nodes[0].Name},
						"placement2": {nodes[1].Name},
					},
				},
				scorePlacementsResult: map[string]map[string]int64{
					pgInfo.GetKey(): {
						"placement1": 1,
					},
				},
				scorePlacementsStatus: map[string]map[string]*fwk.Status{
					pgInfo.GetKey(): {
						"placement2": fwk.NewStatus(fwk.Error, "error for test"),
					},
				},
			},
			expectedResult: podGroupAlgorithmResult{
				status: fwk.NewStatus(fwk.Error, "running PlacementScore plugins: plugin \"FakePlacementPlugin\" failed with: error for test").WithPlugin("FakePlacementPlugin"),
			},
		},
		"when a placement evaluation errors, returns error": {
			expectedGeneratedPlacements: 1,
			placementPlugin: fakePlacementPlugin{
				generatePlacementsResult: map[string]map[string][]string{
					pgInfo.GetKey(): {
						"placement1": {nodes[0].Name},
					},
				},
				scorePlacementsResult: map[string]map[string]int64{
					pgInfo.GetKey(): {
						"placement1": 1,
					},
				},
				filterStatus: map[string]*fwk.Status{
					nodes[0].Name: fwk.NewStatus(fwk.Error, "error for test"),
				},
			},
			expectedResult: podGroupAlgorithmResult{
				podResults: []algorithmResult{
					{
						podInfo: podGroupPodInfo,
						scheduleResult: ScheduleResult{
							nominatingInfo: &fwk.NominatingInfo{NominatingMode: fwk.ModeOverride},
						},
						status: fwk.NewStatus(fwk.Error, "running \"FakePlacementPlugin\" filter plugin: error for test"),
					},
				},
				status: fwk.NewStatus(fwk.Error, "failed to schedule other pod from a pod group: running \"FakePlacementPlugin\" filter plugin: error for test"),
			},
		},
	}
	for _, cpgEnabled := range []bool{false, true} {
		for name, tt := range tests {
			t.Run(fmt.Sprintf("%s (CompositePodGroup=%v)", name, cpgEnabled), func(t *testing.T) {
				featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
					features.TopologyAwareWorkloadScheduling: true,
					features.GenericWorkload:                 true,
					features.CompositePodGroup:               cpgEnabled,
				})

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

				cache := internalcache.New(ctx, nil, true, cpgEnabled /* CompositePodGroup */)
				for _, node := range nodes {
					cache.AddNode(logger, node)
				}
				cache.AddPodGroup(testPodGroup)

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

				metrics.GeneratedPlacementsTotal.Reset()
				metrics.PlacementEvaluations.Reset()
				metrics.PlacementEvaluationDuration.Reset()

				resultsMap := sched.runRootSchedulingAlgorithm(ctx, schedFwk, framework.NewCycleState(), pgInfo)
				result := resultsMap[pgKey(pgInfo.PodGroupInfo)]

				if result.podGroupInfo != pgInfo.PodGroupInfo {
					t.Errorf("Unexpected podGroupInfo field (-want,+got):\n- %v\n+ %v", pgInfo, result.podGroupInfo)
				}

				opts := cmp.Options{
					cmp.AllowUnexported(
						podGroupAlgorithmResult{},
						algorithmResult{},
						ScheduleResult{},
						fwk.Status{},
						framework.PodInfo{}),
					cmp.FilterPath(func(p cmp.Path) bool {
						if len(p) < 2 {
							return false
						}
						step, ok := p[len(p)-1].(cmp.StructField)
						if !ok {
							return false
						}
						return (strings.HasSuffix(p[len(p)-2].Type().String(), "podGroupAlgorithmResult") && (step.Name() == "podGroupInfo" || step.Name() == "placementCycleState" || step.Name() == "revertFn" || step.Name() == "anyScheduled"))
					}, cmp.Ignore()),
					cmpopts.IgnoreFields(algorithmResult{}, "podCtx", "schedulingDuration"),
					statusCmpOpt,
				}

				if diff := cmp.Diff(tt.expectedResult, *result, opts...); diff != "" {
					t.Fatalf("Unexpected algorithm result (-want,+got):\n%s", diff)
				}

				feasibleLabels := map[string]string{"profile": "test-scheduler", "result": metrics.FeasibleResult}
				infeasibleLabels := map[string]string{"profile": "test-scheduler", "result": metrics.InfeasibleResult}
				assertCounterValueFromGatherer(t, testRegistry, "scheduler_generated_placements_total", "profile", "test-scheduler", tt.expectedGeneratedPlacements)
				assertCounterValueFromGatherer(t, testRegistry, "scheduler_placement_evaluations_total", "result", metrics.FeasibleResult, tt.expectedFeasibleEvaluations)
				assertCounterValueFromGatherer(t, testRegistry, "scheduler_placement_evaluations_total", "result", metrics.InfeasibleResult, tt.expectedInfeasibleEvaluations)
				assertHistogramSampleCountFromGatherer(t, testRegistry, "scheduler_placement_evaluation_duration_seconds", feasibleLabels, tt.expectedFeasibleEvaluations)
				assertHistogramSampleCountFromGatherer(t, testRegistry, "scheduler_placement_evaluation_duration_seconds", infeasibleLabels, tt.expectedInfeasibleEvaluations)
			})
		}
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

	for _, cpgEnabled := range []bool{false, true} {
		for name, tt := range tests {
			t.Run(fmt.Sprintf("%s (CompositePodGroup=%v)", name, cpgEnabled), func(t *testing.T) {
				featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
					features.TopologyAwareWorkloadScheduling: true,
					features.GenericWorkload:                 true,
					features.CompositePodGroup:               cpgEnabled,
				})

				logger, ctx := ktesting.NewTestContext(t)

				informerFactory := informers.NewSharedInformerFactory(clientsetfake.NewClientset(), 0)
				queue := internalqueue.NewSchedulingQueue(nil, informerFactory)

				queuedPodInfos := []*framework.QueuedPodInfo{{PodInfo: &framework.PodInfo{Pod: podGroupPod}}}
				pgInfo := &framework.QueuedPodGroupInfo{
					QueuedPodInfos: map[fwk.EntityKey][]*framework.QueuedPodInfo{fwk.MustParseEntityKey("podgroup/default/pg"): queuedPodInfos},
					PodGroupInfo: &framework.PodGroupInfo{
						Name:            "pg",
						Namespace:       "default",
						Type:            fwk.PodGroupKeyType,
						UnscheduledPods: []*v1.Pod{podGroupPod},
					},
				}

				placementPlugin := fakePlacementPlugin{
					name: "FakeGeneratorPlugin",
					generatePlacementsResult: map[string]map[string][]string{
						pgInfo.GetKey(): placements,
					},
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
						name: fmt.Sprintf("FakeScorePlugin[%d]", i),
						scorePlacementsResult: map[string]map[string]int64{
							pgInfo.GetKey(): placementScorePluginData.scorePlacementResult,
						},
						scorePlacementsStatus: map[string]map[string]*fwk.Status{
							pgInfo.GetKey(): placementScorePluginData.scorePlacementStatus,
						},
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

				cache := internalcache.New(ctx, nil, true, cpgEnabled /* CompositePodGroup */)
				for _, node := range nodes {
					cache.AddNode(logger, node)
				}
				testPodGroup := &schedulingv1beta1.PodGroup{
					ObjectMeta: metav1.ObjectMeta{Name: "pg", Namespace: "default"},
				}
				cache.AddPodGroup(testPodGroup)

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

				result, _ := sched.podGroupSchedulingPlacementAlgorithm(ctx, schedFwk, framework.NewCycleState(), pgInfo.PodGroupInfo, pgInfo)

				expectedHost := placements[tt.expectedPlacement][0]
				actualHost := result.podResults[0].scheduleResult.SuggestedHost
				if expectedHost != actualHost {
					t.Fatalf("Unexpected algorithm result, expected placement %s with node %s, got node %s", tt.expectedPlacement, expectedHost, actualHost)
				}
			})
		}
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
	for _, cpgEnabled := range []bool{false, true} {
		t.Run(fmt.Sprintf("CompositePodGroup=%v", cpgEnabled), func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.TopologyAwareWorkloadScheduling: true,
				features.GenericWorkload:                 true,
				features.CompositePodGroup:               cpgEnabled,
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

			cache := internalcache.New(ctx, nil, true, cpgEnabled /* CompositePodGroup */)
			for _, node := range nodes {
				cache.AddNode(logger, node)
			}
			testPodGroup := &schedulingv1beta1.PodGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "pg", Namespace: "default"},
			}
			cache.AddPodGroup(testPodGroup)

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

			queuedPodInfos := []*framework.QueuedPodInfo{{PodInfo: &framework.PodInfo{Pod: podGroupPod}}}
			pgInfo := &framework.QueuedPodGroupInfo{
				QueuedPodInfos: map[fwk.EntityKey][]*framework.QueuedPodInfo{fwk.MustParseEntityKey("podgroup/default/pg"): queuedPodInfos},
				PodGroupInfo: &framework.PodGroupInfo{
					Name:            "pg",
					Namespace:       "default",
					Type:            fwk.PodGroupKeyType,
					UnscheduledPods: []*v1.Pod{podGroupPod},
				},
			}

			result, _ := sched.podGroupSchedulingPlacementAlgorithm(ctx, schedFwk, framework.NewCycleState(), pgInfo.PodGroupInfo, pgInfo)
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
		})
	}
}

type hierarchyData struct {
	id string
}

func (h *hierarchyData) Clone() fwk.StateData { return &hierarchyData{id: h.id} }

var hierarchyKey fwk.StateKey = "hierarchyTracker"

var _ fwk.FilterPlugin = &multiLevelPlacementStateTracker{}
var _ fwk.PlacementGeneratePlugin = &multiLevelPlacementStateTracker{}
var _ fwk.PlacementScorePlugin = &multiLevelPlacementStateTracker{}
var _ framework.PlacementFeasiblePlugin = &multiLevelPlacementStateTracker{}
var _ fwk.PermitPlugin = &multiLevelPlacementStateTracker{}

type multiLevelPlacementStateTracker struct {
	mu                            sync.Mutex
	placementIndex                int
	placementGenerateTrajectories [][]string
	placementFeasibleTrajectories [][]string
	placementScoreTrajectories    [][]string
	filterTrajectories            [][]string
	generatePlacementsResult      map[string][]string
}

func (p *multiLevelPlacementStateTracker) Name() string {
	return names.GangScheduling
}

func collectHierarchyFromPlacementCycleState(cycleState fwk.PlacementCycleState, results *[]string) error {
	if cycleState == nil {
		return nil
	}

	err := collectHierarchyFromPodGroupCycleState(cycleState.GetPodGroupSchedulingCycle(), results)
	if err != nil {
		return err
	}

	state, err := cycleState.Read(hierarchyKey)
	if err != nil {
		return err
	}
	*results = append(*results, state.(*hierarchyData).id)
	return nil
}

func collectHierarchyFromPodGroupCycleState(cycleState fwk.PodGroupCycleState, results *[]string) error {
	if cycleState == nil {
		return nil
	}

	err := collectHierarchyFromPlacementCycleState(cycleState.GetParentPlacementCycleState(), results)
	if err != nil {
		return err
	}

	state, err := cycleState.Read(hierarchyKey)
	if err != nil {
		return err
	}
	*results = append(*results, state.(*hierarchyData).id)
	return nil
}

func (p *multiLevelPlacementStateTracker) Filter(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeInfo fwk.NodeInfo) *fwk.Status {
	trajectory := []string{}
	if err := collectHierarchyFromPlacementCycleState(state.GetPlacementCycleState(), &trajectory); err != nil {
		return fwk.AsStatus(err)
	}
	p.filterTrajectories = append(p.filterTrajectories, trajectory)
	return nil
}

func (p *multiLevelPlacementStateTracker) Permit(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) (*fwk.Status, time.Duration) {
	return nil, 0
}

func (p *multiLevelPlacementStateTracker) PlacementFeasible(ctx context.Context, state fwk.PlacementCycleState, podGroup fwk.PodGroupInfo, args framework.PlacementProgress) *fwk.Status {
	if args.Scheduled == 0 {
		if podGroup.GetPodGroup() != nil {
			trajectory := []string{}
			if err := collectHierarchyFromPodGroupCycleState(state.GetPodGroupSchedulingCycle(), &trajectory); err != nil {
				return fwk.AsStatus(err)
			}
			p.placementFeasibleTrajectories = append(p.placementFeasibleTrajectories, trajectory)
		}
		p.placementIndex += 1
		state.Write(hierarchyKey, &hierarchyData{id: strconv.Itoa(p.placementIndex)})
	}
	return nil
}

func (p *multiLevelPlacementStateTracker) ScorePlacement(ctx context.Context, state fwk.PlacementCycleState, podGroup fwk.PodGroupInfo, placement *fwk.PodGroupAssignments) (int64, *fwk.Status) {
	if podGroup.GetPodGroup() != nil {
		trajectory := []string{}
		if err := collectHierarchyFromPlacementCycleState(state, &trajectory); err != nil {
			return 0, fwk.AsStatus(err)
		}
		p.mu.Lock()
		defer p.mu.Unlock()
		p.placementScoreTrajectories = append(p.placementScoreTrajectories, trajectory)
	}
	return 0, nil
}

func (p *multiLevelPlacementStateTracker) PlacementScoreExtensions() fwk.PlacementScoreExtensions {
	return nil
}

func (p *multiLevelPlacementStateTracker) GeneratePlacements(ctx context.Context, state fwk.PodGroupCycleState, podGroup fwk.PodGroupInfo, parentPlacement *fwk.Placement) (*fwk.GeneratePlacementsResult, *fwk.Status) {
	if podGroup.GetPodGroup() != nil {
		trajectory := []string{}
		if err := collectHierarchyFromPlacementCycleState(state.GetParentPlacementCycleState(), &trajectory); err != nil {
			return nil, fwk.AsStatus(err)
		}
		p.placementGenerateTrajectories = append(p.placementGenerateTrajectories, trajectory)
	}
	state.Write(hierarchyKey, &hierarchyData{id: podGroup.GetKey()})
	placements := []*fwk.Placement{}
	for _, placementName := range p.generatePlacementsResult[podGroup.GetKey()] {
		placements = append(placements, &fwk.Placement{Name: placementName, Nodes: parentPlacement.Nodes})
	}
	return &fwk.GeneratePlacementsResult{Placements: placements}, nil
}

func TestPlacementCycleStateLifecycle_MultiLevel(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.TopologyAwareWorkloadScheduling: true,
		features.GenericWorkload:                 true,
		features.CompositePodGroup:               true,
	})

	nodes := []*v1.Node{
		st.MakeNode().Name("node1").Obj(),
	}

	rootcpg := st.MakeCompositePodGroup().Name("rootcpg").Obj()
	midcpg := st.MakeCompositePodGroup().Name("midcpg").ParentCompositePodGroup("rootcpg").Obj()
	pg := st.MakePodGroup().Name("pg").ParentCompositePodGroup("midcpg").Obj()
	p1 := st.MakePod().Name("p1").UID("p1").PodGroupName("pg").Obj()

	podInfo1, err := framework.NewPodInfo(p1)
	if err != nil {
		t.Fatalf("Failed to create pod info: %v", err)
	}
	queuedPodInfo1 := &framework.QueuedPodInfo{PodInfo: podInfo1}

	leafPGInfo := &framework.PodGroupInfo{
		Name:            pg.Name,
		Namespace:       pg.Namespace,
		Type:            fwk.PodGroupKeyType,
		PodGroup:        pg,
		UnscheduledPods: []*v1.Pod{p1},
	}
	midPGInfo := &framework.PodGroupInfo{
		Name:              midcpg.Name,
		Namespace:         midcpg.Namespace,
		Type:              fwk.CompositePodGroupKeyType,
		CompositePodGroup: midcpg,
		Children:          []*framework.PodGroupInfo{leafPGInfo},
	}
	rootPGInfo := &framework.PodGroupInfo{
		Name:              rootcpg.Name,
		Namespace:         rootcpg.Namespace,
		Type:              fwk.CompositePodGroupKeyType,
		CompositePodGroup: rootcpg,
		Children:          []*framework.PodGroupInfo{midPGInfo},
	}

	logger, ctx := ktesting.NewTestContext(t)

	informerFactory := informers.NewSharedInformerFactory(clientsetfake.NewClientset(), 0)
	queue := internalqueue.NewSchedulingQueue(nil, informerFactory)

	tracker := &multiLevelPlacementStateTracker{
		generatePlacementsResult: map[string][]string{
			rootPGInfo.GetKey(): {
				"placement1",
				"placement2",
			},
			midPGInfo.GetKey(): {
				"placement1",
				"placement2",
			},
			leafPGInfo.GetKey(): {
				"placement1",
				"placement2",
			},
		},
	}

	registry := []tf.RegisterPluginFunc{
		tf.RegisterPlacementGeneratePlugin(tracker.Name(), func(_ context.Context, _ runtime.Object, h fwk.Handle) (fwk.Plugin, error) {
			return tracker, nil
		}),
		tf.RegisterPlacementScorePlugin(tracker.Name(), func(_ context.Context, _ runtime.Object, h fwk.Handle) (fwk.Plugin, error) {
			return tracker, nil
		}, 1),
		tf.RegisterFilterPlugin(tracker.Name(), func(_ context.Context, _ runtime.Object, h fwk.Handle) (fwk.Plugin, error) {
			return tracker, nil
		}),
		tf.RegisterPermitPlugin(tracker.Name(), func(_ context.Context, _ runtime.Object, h fwk.Handle) (fwk.Plugin, error) {
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

	cache := internalcache.New(ctx, nil, true, true /* CompositePodGroup */)
	for _, node := range nodes {
		cache.AddNode(logger, node)
	}
	cache.AddCompositePodGroup(logger, rootcpg)
	cache.AddCompositePodGroup(logger, midcpg)
	cache.AddPodGroup(pg)

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

	cpgQueuedInfo := &framework.QueuedPodGroupInfo{
		QueuedPodInfos: map[fwk.EntityKey][]*framework.QueuedPodInfo{
			fwk.MustParseEntityKey(leafPGInfo.GetKey()): {queuedPodInfo1},
		},
		PodGroupInfo: rootPGInfo,
	}

	results := sched.runRootSchedulingAlgorithm(ctx, schedFwk, framework.NewCycleState(), cpgQueuedInfo)
	if result, ok := results[pgKey(rootPGInfo)]; !ok || !result.status.IsSuccess() {
		t.Fatalf("Expected success for root pod group, got: %v", result.status)
	}

	expectedLeaf := [][]string{
		{"compositepodgroup//rootcpg", "1", "compositepodgroup//midcpg", "2", "podgroup//pg", "3"},
		{"compositepodgroup//rootcpg", "1", "compositepodgroup//midcpg", "2", "podgroup//pg", "4"},
		{"compositepodgroup//rootcpg", "1", "compositepodgroup//midcpg", "5", "podgroup//pg", "6"},
		{"compositepodgroup//rootcpg", "1", "compositepodgroup//midcpg", "5", "podgroup//pg", "7"},
		{"compositepodgroup//rootcpg", "8", "compositepodgroup//midcpg", "9", "podgroup//pg", "10"},
		{"compositepodgroup//rootcpg", "8", "compositepodgroup//midcpg", "9", "podgroup//pg", "11"},
		{"compositepodgroup//rootcpg", "8", "compositepodgroup//midcpg", "12", "podgroup//pg", "13"},
		{"compositepodgroup//rootcpg", "8", "compositepodgroup//midcpg", "12", "podgroup//pg", "14"},
	}
	// each entry is duplicated because we record it for each leaf placement, which has the same parent
	expectedFeasible := [][]string{
		{"compositepodgroup//rootcpg", "1", "compositepodgroup//midcpg", "2", "podgroup//pg"},
		{"compositepodgroup//rootcpg", "1", "compositepodgroup//midcpg", "2", "podgroup//pg"},
		{"compositepodgroup//rootcpg", "1", "compositepodgroup//midcpg", "5", "podgroup//pg"},
		{"compositepodgroup//rootcpg", "1", "compositepodgroup//midcpg", "5", "podgroup//pg"},
		{"compositepodgroup//rootcpg", "8", "compositepodgroup//midcpg", "9", "podgroup//pg"},
		{"compositepodgroup//rootcpg", "8", "compositepodgroup//midcpg", "9", "podgroup//pg"},
		{"compositepodgroup//rootcpg", "8", "compositepodgroup//midcpg", "12", "podgroup//pg"},
		{"compositepodgroup//rootcpg", "8", "compositepodgroup//midcpg", "12", "podgroup//pg"},
	}
	expectedGenerate := [][]string{
		{"compositepodgroup//rootcpg", "1", "compositepodgroup//midcpg", "2"},
		{"compositepodgroup//rootcpg", "1", "compositepodgroup//midcpg", "5"},
		{"compositepodgroup//rootcpg", "8", "compositepodgroup//midcpg", "9"},
		{"compositepodgroup//rootcpg", "8", "compositepodgroup//midcpg", "12"},
	}

	if diff := cmp.Diff(expectedGenerate, tracker.placementGenerateTrajectories); diff != "" {
		t.Errorf("Unexpected placementGenerateTrajectories (-want,+got)\n%s", diff)
	}
	if diff := cmp.Diff(expectedFeasible, tracker.placementFeasibleTrajectories); diff != "" {
		t.Errorf("Unexpected placementFeasibleTrajectories (-want,+got)\n%s", diff)
	}
	if diff := cmp.Diff(expectedLeaf, tracker.filterTrajectories); diff != "" {
		t.Errorf("Unexpected filterTrajectories (-want,+got)\n%s", diff)
	}
	if diff := cmp.Diff(expectedLeaf, tracker.placementScoreTrajectories, cmpopts.SortSlices(func(a, b []string) bool {
		return strings.Join(a, "|") < strings.Join(b, "|")
	})); diff != "" {
		t.Errorf("Unexpected placementScoreTrajectories (-want,+got)\n%s", diff)
	}
}

func TestCPGSchedulingPlacementAlgorithm(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.TopologyAwareWorkloadScheduling: true,
		features.GenericWorkload:                 true,
		features.CompositePodGroup:               true,
	})

	nodes := []*v1.Node{
		st.MakeNode().Name("node1").Obj(),
		st.MakeNode().Name("node2").Obj(),
		st.MakeNode().Name("node3").Obj(),
		st.MakeNode().Name("node4").Obj(),
	}

	cpg := st.MakeCompositePodGroup().Name("cpg").Obj()
	pg1 := st.MakePodGroup().Name("pg1").ParentCompositePodGroup("cpg").Obj()
	pg2 := st.MakePodGroup().Name("pg2").ParentCompositePodGroup("cpg").Obj()
	p1 := st.MakePod().Name("p1").UID("p1").PodGroupName("pg1").Obj()
	p2 := st.MakePod().Name("p2").UID("p2").PodGroupName("pg2").Obj()

	// make sure pg1 is ordered before pg2
	pg1.CreationTimestamp = metav1.NewTime(time.UnixMilli(1))
	pg2.CreationTimestamp = metav1.NewTime(time.UnixMilli(2))

	podInfo1, err := framework.NewPodInfo(p1)
	if err != nil {
		t.Fatalf("Failed to create pod info 1: %v", err)
	}
	podInfo2, err := framework.NewPodInfo(p2)
	if err != nil {
		t.Fatalf("Failed to create pod info 2: %v", err)
	}
	queuedPodInfo1 := &framework.QueuedPodInfo{PodInfo: podInfo1}
	queuedPodInfo2 := &framework.QueuedPodInfo{PodInfo: podInfo2}

	childPGInfo1 := &framework.PodGroupInfo{
		Name:            pg1.Name,
		Namespace:       pg1.Namespace,
		Type:            fwk.PodGroupKeyType,
		PodGroup:        pg1,
		UnscheduledPods: []*v1.Pod{p1},
	}
	childPGInfo2 := &framework.PodGroupInfo{
		Name:            pg2.Name,
		Namespace:       pg2.Namespace,
		Type:            fwk.PodGroupKeyType,
		PodGroup:        pg2,
		UnscheduledPods: []*v1.Pod{p2},
	}

	rootPGInfo := &framework.PodGroupInfo{
		Name:              cpg.Name,
		Namespace:         cpg.Namespace,
		Type:              fwk.CompositePodGroupKeyType,
		CompositePodGroup: cpg,
		Children:          []*framework.PodGroupInfo{childPGInfo1, childPGInfo2},
	}

	defaultPlacementResults := map[string]map[string][]string{
		rootPGInfo.GetKey(): {
			"placement1": {nodes[0].Name, nodes[1].Name},
			"placement2": {nodes[2].Name, nodes[3].Name},
		},
		childPGInfo1.GetKey(): {
			"placement1": {nodes[0].Name},
			"placement2": {nodes[1].Name},
			"placement3": {nodes[2].Name},
			"placement4": {nodes[3].Name},
		},
		childPGInfo2.GetKey(): {
			"placement1": {nodes[0].Name},
			"placement2": {nodes[1].Name},
			"placement3": {nodes[2].Name},
			"placement4": {nodes[3].Name},
		},
	}

	tests := map[string]struct {
		placementPlugin           fakePlacementPlugin
		placementFeasibleStatuses [][]fwk.Code
		expectedResults           map[string]podGroupAlgorithmResult
	}{
		"respects higher score of parent placement": {
			placementPlugin: fakePlacementPlugin{
				generatePlacementsResult: defaultPlacementResults,
				scorePlacementsResult: map[string]map[string]int64{
					rootPGInfo.GetKey(): {
						"placement1": 1,
						"placement2": 2,
					},
					childPGInfo1.GetKey(): {
						"placement1": 5, // should be disregarded due to parent priority
						"placement2": 5, // should be disregarded due to parent priority
						"placement3": 1,
						"placement4": 2,
					},
					childPGInfo2.GetKey(): {
						"placement1": 5, // should be disregarded due to parent priority
						"placement2": 5, // should be disregarded due to parent priority
						"placement3": 2,
						"placement4": 1,
					},
				},
			},
			expectedResults: map[string]podGroupAlgorithmResult{
				rootPGInfo.GetKey(): {},
				childPGInfo1.GetKey(): {
					podResults: []algorithmResult{
						{
							podInfo: queuedPodInfo1,
							scheduleResult: ScheduleResult{
								SuggestedHost:  nodes[3].Name,
								EvaluatedNodes: 1,
								FeasibleNodes:  1,
							},
						},
					},
				},
				childPGInfo2.GetKey(): {
					podResults: []algorithmResult{
						{
							podInfo: queuedPodInfo2,
							scheduleResult: ScheduleResult{
								SuggestedHost:  nodes[2].Name,
								EvaluatedNodes: 1,
								FeasibleNodes:  1,
							},
						},
					},
				},
			},
		},
		"discards infeasible placements": {
			placementPlugin: fakePlacementPlugin{
				generatePlacementsResult: defaultPlacementResults,
				scorePlacementsResult: map[string]map[string]int64{
					rootPGInfo.GetKey(): {
						"placement1": 2,
						"placement2": 1,
					},
					childPGInfo1.GetKey(): {
						"placement1": 5, // should be disregarded due to pod group infeasibility
						"placement2": 5, // should be disregarded due to pod group infeasibility
						"placement3": 1,
						"placement4": 2,
					},
					childPGInfo2.GetKey(): {
						"placement1": 5, // should be disregarded due to pod group infeasibility
						"placement2": 5, // should be disregarded due to pod group infeasibility
						"placement3": 2,
						"placement4": 1,
					},
				},
			},
			placementFeasibleStatuses: [][]fwk.Code{
				// cpg/placement1 (0, 1, 2 PGs evaluated)
				{fwk.Unschedulable, fwk.Unschedulable, fwk.Unschedulable},
				// success for the remaining placements
			},
			expectedResults: map[string]podGroupAlgorithmResult{
				rootPGInfo.GetKey(): {},
				childPGInfo1.GetKey(): {
					podResults: []algorithmResult{
						{
							podInfo: queuedPodInfo1,
							scheduleResult: ScheduleResult{
								SuggestedHost:  nodes[3].Name,
								EvaluatedNodes: 1,
								FeasibleNodes:  1,
							},
						},
					},
				},
				childPGInfo2.GetKey(): {
					podResults: []algorithmResult{
						{
							podInfo: queuedPodInfo2,
							scheduleResult: ScheduleResult{
								SuggestedHost:  nodes[2].Name,
								EvaluatedNodes: 1,
								FeasibleNodes:  1,
							},
						},
					},
				},
			},
		},
		"returns unschedulable if no pods got scheduled": {
			placementPlugin: fakePlacementPlugin{
				generatePlacementsResult: defaultPlacementResults,
				scorePlacementsResult: map[string]map[string]int64{
					rootPGInfo.GetKey(): {
						"placement1": 2,
						"placement2": 1,
					},
					childPGInfo1.GetKey(): {
						"placement1": 5,
						"placement2": 1,
						"placement3": 1,
						"placement4": 1,
					},
					childPGInfo2.GetKey(): {
						"placement1": 5,
						"placement2": 1,
						"placement3": 1,
						"placement4": 1,
					},
				},
				filterStatus: map[string]*fwk.Status{
					nodes[0].Name: fwk.NewStatus(fwk.Unschedulable, "node1 rejected"),
					nodes[1].Name: fwk.NewStatus(fwk.Unschedulable, "node2 rejected"),
					nodes[2].Name: fwk.NewStatus(fwk.Unschedulable, "node3 rejected"),
					nodes[3].Name: fwk.NewStatus(fwk.Unschedulable, "node4 rejected"),
				},
			},
			expectedResults: map[string]podGroupAlgorithmResult{
				rootPGInfo.GetKey(): {
					status: fwk.NewStatus(fwk.Unschedulable, "pod group is unschedulable"),
				},
				childPGInfo1.GetKey(): {
					podResults: []algorithmResult{
						{
							podInfo: queuedPodInfo1,
							status:  fwk.NewStatus(fwk.Unschedulable, "0/1 nodes are available: 1 node1 rejected."),
						},
					},
				},
				childPGInfo2.GetKey(): {
					podResults: []algorithmResult{
						{
							podInfo: queuedPodInfo2,
							status:  fwk.NewStatus(fwk.Unschedulable, "0/1 nodes are available: 1 node1 rejected."),
						},
					},
				},
			},
		},
		"respects pods already scheduled in sibling pod groups": {
			placementPlugin: fakePlacementPlugin{
				generatePlacementsResult: defaultPlacementResults,
				podPerNode:               true,
				reservedNodes:            sets.New[string](),
				scorePlacementsResult: map[string]map[string]int64{
					rootPGInfo.GetKey(): {
						"placement1": 1,
					},
					// same priorities but only 1 PG fits in a given placement in this case
					childPGInfo1.GetKey(): {
						"placement1": 2,
						"placement2": 1,
					},
					childPGInfo2.GetKey(): {
						"placement1": 2,
						"placement2": 1,
					},
				},
				filterStatus: map[string]*fwk.Status{},
			},
			expectedResults: map[string]podGroupAlgorithmResult{
				rootPGInfo.GetKey(): {},
				childPGInfo1.GetKey(): {
					podResults: []algorithmResult{
						{
							podInfo: queuedPodInfo1,
							scheduleResult: ScheduleResult{
								SuggestedHost:  nodes[0].Name,
								EvaluatedNodes: 1,
								FeasibleNodes:  1,
							},
						},
					},
				},
				childPGInfo2.GetKey(): {
					podResults: []algorithmResult{
						{
							podInfo: queuedPodInfo2,
							status:  fwk.NewStatus(fwk.Unschedulable, "0/1 nodes are available: 1 node1 is already reserved."),
						},
					},
				},
			},
		},
		"when generate plugin fails at CPG, returns error": {
			placementPlugin: fakePlacementPlugin{
				generatePlacementsResult: defaultPlacementResults,
				generatePlacementsStatus: map[string]*fwk.Status{
					rootPGInfo.GetKey(): fwk.AsStatus(fmt.Errorf("injected error")),
				},
			},
			expectedResults: map[string]podGroupAlgorithmResult{
				rootPGInfo.GetKey(): {
					status: fwk.NewStatus(fwk.Error, "injected error"),
				},
			},
		},
		"when generate plugin fails at PG, returns error": {
			placementPlugin: fakePlacementPlugin{
				generatePlacementsResult: defaultPlacementResults,
				generatePlacementsStatus: map[string]*fwk.Status{
					childPGInfo2.GetKey(): fwk.AsStatus(fmt.Errorf("injected error")),
				},
			},
			expectedResults: map[string]podGroupAlgorithmResult{
				rootPGInfo.GetKey(): {
					status: fwk.NewStatus(fwk.Error, "composite pod group evaluation failed due to child error: injected error"),
				},
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
				tf.RegisterReservePlugin(tt.placementPlugin.Name(), func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
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

			cache := internalcache.New(ctx, nil, true, true /* CompositePodGroup */)
			for _, node := range nodes {
				cache.AddNode(logger, node)
			}
			cache.AddCompositePodGroup(logger, cpg)
			cache.AddPodGroup(pg1)
			cache.AddPodGroup(pg2)
			cache.AddPodGroupMember(p1)
			cache.AddPodGroupMember(p2)

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

			cpgInfo := &framework.QueuedPodGroupInfo{
				QueuedPodInfos: map[fwk.EntityKey][]*framework.QueuedPodInfo{
					fwk.MustParseEntityKey(childPGInfo1.GetKey()): {queuedPodInfo1},
					fwk.MustParseEntityKey(childPGInfo2.GetKey()): {queuedPodInfo2},
				},
				PodGroupInfo: rootPGInfo,
			}

			results := sched.runRootSchedulingAlgorithm(ctx, schedFwk, framework.NewCycleState(), cpgInfo)
			gotResults := make(map[string]podGroupAlgorithmResult, len(results))
			for k, v := range results {
				if v != nil {
					gotResults[k.String()] = *v
				}
			}

			opts := cmp.Options{
				cmp.AllowUnexported(
					podGroupAlgorithmResult{},
					algorithmResult{},
					ScheduleResult{},
					fwk.Status{},
					framework.PodInfo{}),
				cmp.FilterPath(func(p cmp.Path) bool {
					if len(p) < 2 {
						return false
					}
					step, ok := p[len(p)-1].(cmp.StructField)
					if !ok {
						return false
					}
					return (strings.HasSuffix(p[len(p)-2].Type().String(), "podGroupAlgorithmResult") && (step.Name() == "podGroupInfo" || step.Name() == "placementCycleState" || step.Name() == "revertFn" || step.Name() == "anyScheduled"))
				}, cmp.Ignore()),
				cmpopts.IgnoreFields(algorithmResult{}, "podCtx", "schedulingDuration"),
				statusCmpOpt,
			}

			if diff := cmp.Diff(tt.expectedResults, gotResults, opts...); diff != "" {
				t.Fatalf("Unexpected algorithm results (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestCPGSchedulingPlacementAlgorithm_Scoring(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.TopologyAwareWorkloadScheduling: true,
		features.GenericWorkload:                 true,
		features.CompositePodGroup:               true,
	})

	nodes := []*v1.Node{
		st.MakeNode().Name("node1").Obj(),
		st.MakeNode().Name("node2").Obj(),
		st.MakeNode().Name("node3").Obj(),
		st.MakeNode().Name("node4").Obj(),
	}
	p1 := st.MakePod().Name("p1").UID("p1").PodGroupName("pg1").Obj()
	p2 := st.MakePod().Name("p2").UID("p2").PodGroupName("pg2").Obj()

	cpg := st.MakeCompositePodGroup().Name("cpg").Obj()
	pg1 := st.MakePodGroup().Name("pg1").Obj()
	pg2 := st.MakePodGroup().Name("pg2").Obj()

	podInfo1, err := framework.NewPodInfo(p1)
	if err != nil {
		t.Fatalf("Failed to create pod info 1: %v", err)
	}
	podInfo2, err := framework.NewPodInfo(p2)
	if err != nil {
		t.Fatalf("Failed to create pod info 2: %v", err)
	}
	queuedPodInfo1 := &framework.QueuedPodInfo{PodInfo: podInfo1}
	queuedPodInfo2 := &framework.QueuedPodInfo{PodInfo: podInfo2}

	childPGInfo1 := &framework.PodGroupInfo{
		Name:            pg1.Name,
		Namespace:       pg1.Namespace,
		Type:            fwk.PodGroupKeyType,
		PodGroup:        pg1,
		UnscheduledPods: []*v1.Pod{p1},
	}
	childPGInfo2 := &framework.PodGroupInfo{
		Name:            pg2.Name,
		Namespace:       pg2.Namespace,
		Type:            fwk.PodGroupKeyType,
		PodGroup:        pg2,
		UnscheduledPods: []*v1.Pod{p2},
	}

	rootPGInfo := &framework.PodGroupInfo{
		Name:              cpg.Name,
		Namespace:         cpg.Namespace,
		Type:              fwk.CompositePodGroupKeyType,
		CompositePodGroup: cpg,
		Children:          []*framework.PodGroupInfo{childPGInfo1, childPGInfo2},
	}

	placements := map[string]map[string][]string{
		rootPGInfo.GetKey(): {
			"placement1": {nodes[0].Name, nodes[1].Name},
			"placement2": {nodes[2].Name, nodes[3].Name},
		},
		childPGInfo1.GetKey(): {
			"placement1": {nodes[0].Name},
			"placement2": {nodes[1].Name},
			"placement3": {nodes[2].Name},
			"placement4": {nodes[3].Name},
		},
		childPGInfo2.GetKey(): {
			"placement1": {nodes[0].Name},
			"placement2": {nodes[1].Name},
			"placement3": {nodes[2].Name},
			"placement4": {nodes[3].Name},
		},
	}

	type pluginData struct {
		weight                int32
		scorePlacementsResult map[string]map[string]int64
		scorePlacementsStatus map[string]map[string]*fwk.Status
	}

	tests := map[string]struct {
		pluginData    []pluginData
		expectedHosts map[string]string
	}{
		"respects higher score of root placement1": {
			pluginData: []pluginData{
				{
					weight: 1,
					scorePlacementsResult: map[string]map[string]int64{
						rootPGInfo.GetKey(): {
							"placement1": 50,
							"placement2": 75,
						},
						childPGInfo1.GetKey(): {
							"placement1": 10,
							"placement2": 5,
							"placement3": 10,
							"placement4": 5,
						},
						childPGInfo2.GetKey(): {
							"placement1": 5,
							"placement2": 10,
							"placement3": 5,
							"placement4": 10,
						},
					},
				},
				{
					weight: 2,
					scorePlacementsResult: map[string]map[string]int64{
						rootPGInfo.GetKey(): {
							"placement1": 25,
							"placement2": 10,
						},
						childPGInfo1.GetKey(): {
							"placement1": 10,
							"placement2": 5,
							"placement3": 10,
							"placement4": 5,
						},
						childPGInfo2.GetKey(): {
							"placement1": 5,
							"placement2": 10,
							"placement3": 5,
							"placement4": 10,
						},
					},
				},
			},
			expectedHosts: map[string]string{
				p1.Name: nodes[0].Name,
				p2.Name: nodes[1].Name,
			},
		},
		"respects higher score of root placement2": {
			pluginData: []pluginData{
				{
					weight: 1,
					scorePlacementsResult: map[string]map[string]int64{
						rootPGInfo.GetKey(): {
							"placement1": 75,
							"placement2": 50,
						},
						childPGInfo1.GetKey(): {
							"placement1": 10,
							"placement2": 5,
							"placement3": 10,
							"placement4": 5,
						},
						childPGInfo2.GetKey(): {
							"placement1": 5,
							"placement2": 10,
							"placement3": 5,
							"placement4": 10,
						},
					},
				},
				{
					weight: 2,
					scorePlacementsResult: map[string]map[string]int64{
						rootPGInfo.GetKey(): {
							"placement1": 10,
							"placement2": 25,
						},
						childPGInfo1.GetKey(): {
							"placement1": 10,
							"placement2": 5,
							"placement3": 10,
							"placement4": 5,
						},
						childPGInfo2.GetKey(): {
							"placement1": 5,
							"placement2": 10,
							"placement3": 5,
							"placement4": 10,
						},
					},
				},
			},
			expectedHosts: map[string]string{
				p1.Name: nodes[2].Name,
				p2.Name: nodes[3].Name,
			},
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

			orderedPlacementGeneratePlugin := &orderedPlacementPlugin{&placementPlugin}
			gangPluginFactory := func(ctx context.Context, obj runtime.Object, handle fwk.Handle) (fwk.Plugin, error) {
				return gangscheduling.New(ctx, obj, handle, feature.Features{EnableTopologyAwareWorkloadScheduling: true})
			}

			registry := []tf.RegisterPluginFunc{
				tf.RegisterPlacementGeneratePlugin(orderedPlacementGeneratePlugin.Name(), func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
					return orderedPlacementGeneratePlugin, nil
				}),
				tf.RegisterFilterPlugin(placementPlugin.Name(), func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
					return &placementPlugin, nil
				}),
				tf.RegisterPermitPlugin(gangscheduling.Name, gangPluginFactory),
				tf.RegisterPluginAsExtensions(gangscheduling.Name, gangPluginFactory, "PlacementFeasible"),
			}

			for i, placementScorePluginData := range tt.pluginData {
				plugin := fakePlacementPlugin{
					name:                  fmt.Sprintf("FakeScorePlugin[%d]", i),
					scorePlacementsResult: placementScorePluginData.scorePlacementsResult,
					scorePlacementsStatus: placementScorePluginData.scorePlacementsStatus,
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

			cache := internalcache.New(ctx, nil, true, true /* CompositePodGroup */)
			for _, node := range nodes {
				cache.AddNode(logger, node)
			}
			cache.AddCompositePodGroup(logger, cpg)
			cache.AddPodGroup(pg1)
			cache.AddPodGroup(pg2)
			cache.AddPodGroupMember(p1)
			cache.AddPodGroupMember(p2)

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

			cpgInfo := &framework.QueuedPodGroupInfo{
				QueuedPodInfos: map[fwk.EntityKey][]*framework.QueuedPodInfo{
					fwk.MustParseEntityKey(childPGInfo1.GetKey()): {queuedPodInfo1},
					fwk.MustParseEntityKey(childPGInfo2.GetKey()): {queuedPodInfo2},
				},
				PodGroupInfo: rootPGInfo,
			}

			results := sched.runRootSchedulingAlgorithm(ctx, schedFwk, framework.NewCycleState(), cpgInfo)
			gotHosts := make(map[string]string)
			for _, result := range results {
				for _, pr := range result.podResults {
					gotHosts[pr.podInfo.Pod.Name] = pr.scheduleResult.SuggestedHost
				}
			}

			if diff := cmp.Diff(tt.expectedHosts, gotHosts); diff != "" {
				t.Fatalf("Unexpected suggested hosts (-want,+got):\n%s", diff)
			}
		})
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
		QueuedPodInfos: map[fwk.EntityKey][]*framework.QueuedPodInfo{fwk.MustParseEntityKey("podgroup/default/pg"): {qInfo1, qInfo2}},
		PodGroupInfo: &framework.PodGroupInfo{
			Name:            "pg",
			Namespace:       "default",
			Type:            fwk.PodGroupKeyType,
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

	cache := internalcache.New(ctx, nil, true, true /* CompositePodGroup */)
	cache.AddPodGroup(testPodGroup)
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

	if err := sched.Cache.UpdateSnapshot(logger, sched.nodeInfoSnapshot); err != nil {
		t.Fatalf("Failed to update snapshot: %v", err)
	}
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

	testPodGroup := st.MakePodGroup().Name("pg").Namespace("default").Obj()
	p1 := st.MakePod().Name("p1").Namespace("default").UID("p1").PodGroupName("pg").SchedulerName("test-scheduler").Obj()
	p2 := st.MakePod().Name("p2").Namespace("default").UID("p2").PodGroupName("pg").SchedulerName("test-scheduler").Obj()
	qInfo1 := &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: p1}}
	qInfo2 := &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: p2}}

	podGroupInfo := &framework.QueuedPodGroupInfo{
		QueuedPodInfos: map[fwk.EntityKey][]*framework.QueuedPodInfo{fwk.MustParseEntityKey("podgroup/default/pg"): {qInfo1, qInfo2}},
		PodGroupInfo: &framework.PodGroupInfo{
			Name:            "pg",
			Namespace:       "default",
			Type:            fwk.PodGroupKeyType,
			PodGroup:        testPodGroup,
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

	cache := internalcache.New(ctx, nil, true, false /* CompositePodGroup */)
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
		QueuedPodInfos: map[fwk.EntityKey][]*framework.QueuedPodInfo{fwk.MustParseEntityKey("podgroup/default/pg"): {qInfo1, qInfo2}},
		PodGroupInfo: &framework.PodGroupInfo{
			Name:      "pg",
			Namespace: "default",
			Type:      fwk.PodGroupKeyType,
			PodGroup:  testPodGroup,
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
			Cache: internalcache.New(ctx, nil, true, false /* CompositePodGroup */),
			UpdateSnapshotFunc: func(nodeSnapshot *internalcache.Snapshot) error {
				return nil
			},
		},
		nodeInfoSnapshot: internalcache.NewTestSnapshotWithPodGroups(
			[]*v1.Pod{st.MakePod().Name("p").Namespace("default").UID("p").PodGroupName("pg").Node("node1").SchedulerName("sched1").Obj()},
			[]*v1.Node{st.MakeNode().Name("node1").Obj()},
			[]*schedulingv1beta1.PodGroup{st.MakePodGroup().Name("pg").Namespace("default").UID("pg").Obj()},
		),
		client: client,
		FailureHandler: func(ctx context.Context, fwk framework.Framework, p *framework.QueuedPodInfo, status *fwk.Status, ni *fwk.NominatingInfo, start time.Time) {
		},
	}

	sched.scheduleOnePodGroup(ctx, podGroupInfo)

	pg, err := client.SchedulingV1beta1().PodGroups("default").Get(ctx, "pg", metav1.GetOptions{})
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

func TestCPGHierarchicalScheduling_ScheduleOnePodGroup(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.CompositePodGroup:               true,
		features.GenericWorkload:                 true,
		features.TopologyAwareWorkloadScheduling: true,
	})

	// Tree structure:
	//                  cpg-root (Gang, MinGroupCount: 2)
	//                  /       |       \
	//                pg1      pg2      pg3
	//               (Min:1)  (Min:1)  (Min:1)
	//
	// We will schedule cpg-root.
	// Since MinGroupCount: 2, if 2 out of 3 child pod groups are schedulable, it should succeed!

	namespace := "default"

	cpgRoot := &schedulingv1alpha3.CompositePodGroup{
		ObjectMeta: metav1.ObjectMeta{Namespace: namespace, Name: "cpg-root"},
		Spec: schedulingv1alpha3.CompositePodGroupSpec{
			SchedulingPolicy: schedulingv1alpha3.CompositePodGroupSchedulingPolicy{
				Gang: &schedulingv1alpha3.CompositeGangSchedulingPolicy{MinGroupCount: 2},
			},
		},
	}

	pg1 := st.MakePodGroup().Name("pg1").Namespace(namespace).ParentCompositePodGroup("cpg-root").MinCount(1).Obj()
	pg2 := st.MakePodGroup().Name("pg2").Namespace(namespace).ParentCompositePodGroup("cpg-root").MinCount(1).Obj()
	pg3 := st.MakePodGroup().Name("pg3").Namespace(namespace).ParentCompositePodGroup("cpg-root").MinCount(1).Obj()

	p1 := st.MakePod().Name("p1").UID("p1").PodGroupName("pg1").SchedulerName("test-scheduler").Obj()
	p2 := st.MakePod().Name("p2").UID("p2").PodGroupName("pg2").SchedulerName("test-scheduler").Obj()
	p3 := st.MakePod().Name("p3").UID("p3").PodGroupName("pg3").SchedulerName("test-scheduler").Obj()

	qInfo1 := &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: p1}}
	qInfo2 := &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: p2}}
	qInfo3 := &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: p3}}
	queuedPodInfosMap := map[fwk.EntityKey][]*framework.QueuedPodInfo{
		fwk.MustParseEntityKey("podgroup/default/pg1"): {qInfo1},
		fwk.MustParseEntityKey("podgroup/default/pg2"): {qInfo2},
		fwk.MustParseEntityKey("podgroup/default/pg3"): {qInfo3},
	}

	cpgRootInfo := &framework.QueuedPodGroupInfo{
		QueuedPodInfos: queuedPodInfosMap,
		PodGroupInfo: &framework.PodGroupInfo{
			Namespace:         namespace,
			Name:              "cpg-root",
			Type:              fwk.CompositePodGroupKeyType,
			CompositePodGroup: cpgRoot,
			UnscheduledPods:   []*v1.Pod{p1, p2, p3},
			Children: []*framework.PodGroupInfo{
				{
					Name:            "pg1",
					Namespace:       "default",
					Type:            fwk.PodGroupKeyType,
					PodGroup:        pg1,
					UnscheduledPods: []*v1.Pod{p1},
				},
				{
					Name:            "pg2",
					Namespace:       "default",
					Type:            fwk.PodGroupKeyType,
					PodGroup:        pg2,
					UnscheduledPods: []*v1.Pod{p2},
				},
				{
					Name:            "pg3",
					Namespace:       "default",
					Type:            fwk.PodGroupKeyType,
					PodGroup:        pg3,
					UnscheduledPods: []*v1.Pod{p3},
				},
			},
		},
		QueueingParams: framework.QueueingParams{
			Timestamp: time.Now(),
		},
	}

	logger, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	// Mock PlacementFeasible plugin
	fakeGS := &fakePlacementFeasiblePlugin{}
	fakeGS.placementFeasibleStatuses = [][]fwk.Code{
		// Four distinct scheduling cycles:
		// 1. cpg-root (success, 1 call: before children scheduling, evaluated=0)
		// 2. pg1 scheduling (success, 2 calls: before and after p1, evaluated=0 and 1)
		// 3. pg2 scheduling (success, 2 calls: before and after p2, evaluated=0 and 1)
		// 4. pg3 scheduling (fails early on Evaluated=0)
		{fwk.Success},
		{fwk.Success, fwk.Success},
		{fwk.Success, fwk.Success, fwk.Success},
		{fwk.Unschedulable},
	}

	registry := frameworkruntime.Registry{
		queuesort.Name:     queuesort.New,
		defaultbinder.Name: defaultbinder.New,
		names.GangScheduling: func(ctx context.Context, obj runtime.Object, handle fwk.Handle) (fwk.Plugin, error) {
			return fakeGS, nil
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
			// Enable GangScheduling to run PlacementFeasible
			Permit: config.PluginSet{
				Enabled: []config.Plugin{{Name: names.GangScheduling}},
			},
		},
	}

	client := clientsetfake.NewSimpleClientset(cpgRoot, pg1, pg2, pg3)
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

	cache := internalcache.New(ctx, nil, true, true /* CompositePodGroup */)
	cache.AddCompositePodGroup(logger, cpgRoot)
	cache.AddPodGroup(pg1)
	cache.AddPodGroup(pg2)
	cache.AddPodGroup(pg3)

	sched := &Scheduler{
		Profiles:         profile.Map{"test-scheduler": schedFwk},
		Cache:            cache,
		nodeInfoSnapshot: internalcache.NewEmptySnapshot(),
		client:           client,
		SchedulingQueue:  internalqueue.NewTestQueue(ctx, nil),
	}

	// Mock SchedulePod to return success for all pods
	sched.SchedulePod = func(ctx context.Context, fwk framework.Framework, state fwk.CycleState, podInfo *framework.QueuedPodInfo) (ScheduleResult, error) {
		return ScheduleResult{SuggestedHost: "node1"}, nil
	}

	// Prepare nodeInfoSnapshot which matches Cache
	if err := sched.Cache.UpdateSnapshot(logger, sched.nodeInfoSnapshot); err != nil {
		t.Fatalf("Failed to update snapshot: %v", err)
	}

	// Run podGroupSchedulingRecursiveAlgorithm
	res := map[fwk.EntityKey]*podGroupAlgorithmResult{}
	result, _ := sched.podGroupSchedulingRecursiveAlgorithm(ctx, schedFwk, framework.NewCycleState(), cpgRootInfo, cpgRootInfo.PodGroupInfo, res)

	status := result.status
	if status.Code() != fwk.Success {
		t.Errorf("Expected recursive scheduling algorithm status to be Success, got: %v", status)
	}

	// Verify that exactly the successful child pod groups (pg1 and pg2) are in the scheduled pod results
	// pg3 should NOT be in the results because its placement was not feasible!
	var successPodNames []string
	for _, childRes := range res {
		if childRes.podGroupInfo.CompositePodGroup != nil {
			continue
		}
		for _, pr := range childRes.podResults {
			if childRes.status.IsSuccess() && pr.status.IsSuccess() {
				successPodNames = append(successPodNames, pr.podInfo.Pod.Name)
			}
		}
	}

	scheduledPodNames := sets.New(successPodNames...)
	if scheduledPodNames.Len() != 2 {
		t.Errorf("Expected 2 scheduled pod results, got %d", scheduledPodNames.Len())
	}

	if !scheduledPodNames.Has("p1") || !scheduledPodNames.Has("p2") {
		t.Errorf("Expected p1 and p2 to be scheduled, got: %v", scheduledPodNames.UnsortedList())
	}
	if scheduledPodNames.Has("p3") {
		t.Errorf("Expected p3 NOT to be scheduled, but got in results")
	}
}

// TestCPGHierarchicalScheduling_Internal tests a complex, multi-level hierarchy of CompositePodGroups.
// It verifies that Gang and Basic scheduling semantics apply correctly at each level.
//
// Tree structure:
//
//	                   cpg-root (Gang, MinGroup: 2)
//	                /               |                \
//	      cpg-sub1               cpg-sub2             cpg-sub3
//	(Gang, MinGroup: 2)          (Basic)         (Gang, MinGroup: 2)
//	    /    |    \               /     \              /     \
//	  pg1   pg2   pg3           pg4     pg5          pg6     pg7
//	  (S)   (S)   (F)           (S)     (S)          (S->F)  (F)
//
// (S) = Success
// (F) = Fail (filter unschedulable)
// (S->F) = Success initially, but fails because its parent cpg-sub3 (MinGroup: 2) fails.
//
// Results:
// - cpg-sub1 succeeds (2 out of 3 groups succeed: pg1, pg2).
// - cpg-sub2 succeeds (Basic, so pg4 and pg5 succeed independently).
// - cpg-sub3 fails (only 1 group, pg6, succeeds, which is < MinGroup: 2. pg7 fails).
// - cpg-root succeeds (2 out of 3 sub-groups succeed: cpg-sub1, cpg-sub2).
func TestCPGHierarchicalScheduling_Internal(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.CompositePodGroup:               true,
		features.GenericWorkload:                 true,
		features.TopologyAwareWorkloadScheduling: true,
	})

	testNode := st.MakeNode().Name("node1").UID("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "100", v1.ResourceMemory: "100Gi", v1.ResourcePods: "100"}).Obj()
	testNode.Status.Allocatable = testNode.Status.Capacity

	logger, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	cache := internalcache.New(ctx, nil, true, true /* CompositePodGroup */)
	cache.AddNode(logger, testNode)

	ns := "default"

	// Helper function to create PodGroups
	createPG := func(name string, minCount int32, parentCPG string) *schedulingv1beta1.PodGroup {
		var policy schedulingv1beta1.PodGroupSchedulingPolicy
		if minCount > 0 {
			policy = schedulingv1beta1.PodGroupSchedulingPolicy{
				Gang: &schedulingv1beta1.GangSchedulingPolicy{MinCount: minCount},
			}
		} else {
			policy = schedulingv1beta1.PodGroupSchedulingPolicy{
				Basic: &schedulingv1beta1.BasicSchedulingPolicy{},
			}
		}
		pg := &schedulingv1beta1.PodGroup{
			ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: ns},
			Spec: schedulingv1beta1.PodGroupSpec{
				SchedulingPolicy:            policy,
				ParentCompositePodGroupName: new(parentCPG),
			},
		}
		cache.AddPodGroup(pg)
		return pg
	}

	createCPG := func(name string, minCount int32, parentCPG *string) *schedulingv1alpha3.CompositePodGroup {
		var policy schedulingv1alpha3.CompositePodGroupSchedulingPolicy
		if minCount > 0 {
			policy = schedulingv1alpha3.CompositePodGroupSchedulingPolicy{
				Gang: &schedulingv1alpha3.CompositeGangSchedulingPolicy{MinGroupCount: minCount},
			}
		} else {
			policy = schedulingv1alpha3.CompositePodGroupSchedulingPolicy{
				Basic: &schedulingv1alpha3.CompositeBasicSchedulingPolicy{},
			}
		}
		cpg := &schedulingv1alpha3.CompositePodGroup{
			ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: ns},
			Spec: schedulingv1alpha3.CompositePodGroupSpec{
				SchedulingPolicy:            policy,
				ParentCompositePodGroupName: parentCPG,
			},
		}
		cache.AddCompositePodGroup(logger, cpg)
		return cpg
	}

	rootCPG := createCPG("cpg-root", 2, nil)
	cpgSub1 := createCPG("cpg-sub1", 2, new("cpg-root"))
	cpgSub2 := createCPG("cpg-sub2", 0, new("cpg-root"))
	cpgSub3 := createCPG("cpg-sub3", 2, new("cpg-root"))

	pg1 := createPG("pg1", 3, "cpg-sub1")
	pg2 := createPG("pg2", 3, "cpg-sub1")
	pg3 := createPG("pg3", 3, "cpg-sub1")

	pg4 := createPG("pg4", 0, "cpg-sub2")
	pg5 := createPG("pg5", 3, "cpg-sub2")

	pg6 := createPG("pg6", 3, "cpg-sub3")
	pg7 := createPG("pg7", 3, "cpg-sub3")

	var allPods []*v1.Pod
	queuedPodInfos := make(map[fwk.EntityKey][]*framework.QueuedPodInfo)
	pgPods := make(map[string][]*v1.Pod)

	createPods := func(pgName string, count int) {
		for i := range count {
			pod := st.MakePod().Namespace(ns).Name(fmt.Sprintf("%s-pod-%d", pgName, i)).
				UID(fmt.Sprintf("%s-pod-%d", pgName, i)).
				PodGroupName(pgName).Priority(100).SchedulerName("test-scheduler").Obj()

			allPods = append(allPods, pod)
			key := fwk.PodGroupKey(ns, pgName)
			queuedPodInfos[key] = append(queuedPodInfos[key], &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: pod}})
			pgPods[pgName] = append(pgPods[pgName], pod)
			cache.AddPodGroupMember(pod)
		}
	}

	createPods("pg1", 3)
	createPods("pg2", 3)
	createPods("pg3", 3)
	createPods("pg4", 3)
	createPods("pg5", 3)
	createPods("pg6", 3)
	createPods("pg7", 3)

	podGroupInfo := &framework.QueuedPodGroupInfo{
		QueuedPodInfos: queuedPodInfos,
		PodGroupInfo: &framework.PodGroupInfo{
			Namespace:         ns,
			Name:              "cpg-root",
			Type:              fwk.CompositePodGroupKeyType,
			UnscheduledPods:   allPods,
			CompositePodGroup: rootCPG,
			Children: []*framework.PodGroupInfo{
				{
					Namespace:         ns,
					Name:              "cpg-sub1",
					Type:              fwk.CompositePodGroupKeyType,
					CompositePodGroup: cpgSub1,
					Children: []*framework.PodGroupInfo{
						{Namespace: ns, Name: "pg1", Type: fwk.PodGroupKeyType, PodGroup: pg1, UnscheduledPods: pgPods["pg1"]},
						{Namespace: ns, Name: "pg2", Type: fwk.PodGroupKeyType, PodGroup: pg2, UnscheduledPods: pgPods["pg2"]},
						{Namespace: ns, Name: "pg3", Type: fwk.PodGroupKeyType, PodGroup: pg3, UnscheduledPods: pgPods["pg3"]},
					},
				},
				{
					Namespace:         ns,
					Name:              "cpg-sub2",
					Type:              fwk.CompositePodGroupKeyType,
					CompositePodGroup: cpgSub2,
					Children: []*framework.PodGroupInfo{
						{Namespace: ns, Name: "pg4", Type: fwk.PodGroupKeyType, PodGroup: pg4, UnscheduledPods: pgPods["pg4"]},
						{Namespace: ns, Name: "pg5", Type: fwk.PodGroupKeyType, PodGroup: pg5, UnscheduledPods: pgPods["pg5"]},
					},
				},
				{
					Namespace:         ns,
					Name:              "cpg-sub3",
					Type:              fwk.CompositePodGroupKeyType,
					CompositePodGroup: cpgSub3,
					Children: []*framework.PodGroupInfo{
						{Namespace: ns, Name: "pg6", Type: fwk.PodGroupKeyType, PodGroup: pg6, UnscheduledPods: pgPods["pg6"]},
						{Namespace: ns, Name: "pg7", Type: fwk.PodGroupKeyType, PodGroup: pg7, UnscheduledPods: pgPods["pg7"]},
					},
				},
			},
		},
	}

	fakePlugin := &fakePodGroupPlugin{
		filterStatus: map[string]*fwk.Status{
			"pg3-pod-0": fwk.NewStatus(fwk.Unschedulable),
			"pg3-pod-1": fwk.NewStatus(fwk.Unschedulable),
			"pg3-pod-2": fwk.NewStatus(fwk.Unschedulable),
			"pg7-pod-0": fwk.NewStatus(fwk.Unschedulable),
			"pg7-pod-1": fwk.NewStatus(fwk.Unschedulable),
			"pg7-pod-2": fwk.NewStatus(fwk.Unschedulable),
			"pg1-pod-0": nil,
			"pg1-pod-1": nil,
			"pg1-pod-2": nil,
			"pg2-pod-0": nil,
			"pg2-pod-1": nil,
			"pg2-pod-2": nil,
			"pg4-pod-0": nil,
			"pg4-pod-1": nil,
			"pg4-pod-2": nil,
			"pg5-pod-0": nil,
			"pg5-pod-1": nil,
			"pg5-pod-2": nil,
			"pg6-pod-0": nil,
			"pg6-pod-1": nil,
			"pg6-pod-2": nil,
		},
	}

	gangPluginFactory := func(ctx context.Context, obj runtime.Object, handle fwk.Handle) (fwk.Plugin, error) {
		return gangscheduling.New(ctx, obj, handle, feature.Features{EnableTopologyAwareWorkloadScheduling: true, EnableCompositePodGroup: true})
	}

	registry := []tf.RegisterPluginFunc{
		tf.RegisterFilterPlugin(fakePlugin.Name(), func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
			return fakePlugin, nil
		}),
		tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
		tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
		tf.RegisterPermitPlugin(gangscheduling.Name, gangPluginFactory),
		tf.RegisterPluginAsExtensions(gangscheduling.Name, gangPluginFactory, "PlacementFeasible"),
	}

	clientObjs := []runtime.Object{testNode, rootCPG, cpgSub1, cpgSub2, cpgSub3, pg1, pg2, pg3, pg4, pg5, pg6, pg7}
	for _, pod := range allPods {
		clientObjs = append(clientObjs, pod)
	}
	client := clientsetfake.NewSimpleClientset(clientObjs...)
	informerFactory := informers.NewSharedInformerFactory(client, 0)

	informerFactory.Start(ctx.Done())
	informerFactory.WaitForCacheSync(ctx.Done())
	queue := internalqueue.NewSchedulingQueue(nil, informerFactory)
	snapshot := internalcache.NewEmptySnapshot()

	schedFwk, err := tf.NewFramework(ctx, registry, "test-scheduler",
		frameworkruntime.WithInformerFactory(informerFactory),
		frameworkruntime.WithSnapshotSharedLister(snapshot),
		frameworkruntime.WithPodNominator(queue),
		frameworkruntime.WithPodActivator(queue),
		frameworkruntime.WithClientSet(client),
		frameworkruntime.WithEventRecorder(events.NewFakeRecorder(100)),
		frameworkruntime.WithWaitingPods(frameworkruntime.NewWaitingPodsMap()),
		frameworkruntime.WithPodsInPreBind(frameworkruntime.NewPodsInPreBindMap()),
		frameworkruntime.WithPodGroupManager(cache),
	)
	informerFactory.Start(ctx.Done())
	informerFactory.WaitForCacheSync(ctx.Done())
	if err != nil {
		t.Fatalf("Failed to create new framework: %v", err)
	}

	handledPods := make(map[string]*fwk.Status)
	var lock sync.Mutex

	sched := &Scheduler{
		Profiles:         profile.Map{"test-scheduler": schedFwk},
		SchedulingQueue:  internalqueue.NewTestQueue(ctx, nil),
		Cache:            cache,
		client:           client,
		nodeInfoSnapshot: snapshot, // will be updated by UpdateSnapshot
		FailureHandler: func(ctx context.Context, fwk framework.Framework, p *framework.QueuedPodInfo, status *fwk.Status, ni *fwk.NominatingInfo, start time.Time) {
			lock.Lock()
			defer lock.Unlock()
			handledPods[p.Pod.Name] = status
		},
	}
	sched.SchedulePod = sched.schedulePod

	// Run the scheduling cycle
	if err := cache.UpdateSnapshot(logger, snapshot); err != nil {
		t.Fatalf("Failed to update snapshot: %v", err)
	}
	t.Logf("Node info list size: %d", func() int { l, _ := snapshot.NodeInfos().List(); return len(l) }())
	podGroupCycleState := framework.NewCycleState()
	podGroupCycleState.SetPodGroupSchedulingCycle(podGroupCycleState)
	sched.podGroupCycle(ctx, schedFwk, podGroupCycleState, podGroupInfo, time.Now())

	lock.Lock()
	defer lock.Unlock()

	// Verify the results
	successPods := []string{
		"pg1-pod-0", "pg1-pod-1", "pg1-pod-2",
		"pg2-pod-0", "pg2-pod-1", "pg2-pod-2",
		"pg4-pod-0", "pg4-pod-1", "pg4-pod-2",
		"pg5-pod-0", "pg5-pod-1", "pg5-pod-2",
	}

	failPods := []string{
		"pg3-pod-0", "pg3-pod-1", "pg3-pod-2",
		"pg6-pod-0", "pg6-pod-1", "pg6-pod-2",
		"pg7-pod-0", "pg7-pod-1", "pg7-pod-2",
	}

	for _, p := range successPods {
		if status, ok := handledPods[p]; ok {
			t.Errorf("Expected pod %s to be scheduled successfully, but got error: %v", p, status.AsError())
		}
	}

	for _, p := range failPods {
		if _, ok := handledPods[p]; !ok {
			t.Errorf("Expected pod %s to fail, but it was scheduled successfully", p)
		}
	}
}

// TestCPGMinGroupCount_Internal verifies that a Gang CompositePodGroup schedules its children
// if at least MinGroupCount of them are fully schedulable.
//
// Tree structure:
//
//	   cpg-root (Gang, MinGroup: 2)
//	  /             |              \
//	pg1            pg2            pg3
//	(S)            (S)            (F)
//
// (S) = Success
// (F) = Fail
func TestCPGMinGroupCount_Internal(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.CompositePodGroup:               true,
		features.GenericWorkload:                 true,
		features.TopologyAwareWorkloadScheduling: true,
	})

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	logger := klog.FromContext(ctx)

	// Capacity matches 2 groups. pg3 pods fail.
	testNode := st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "20"}).Obj()
	cache := internalcache.New(ctx, nil, true, true /* CompositePodGroup */)
	cache.AddNode(logger, testNode)

	ns := "default"

	createCPG := func(name string, minCount int32, parentCPG *string) *schedulingv1alpha3.CompositePodGroup {
		var policy schedulingv1alpha3.CompositePodGroupSchedulingPolicy
		if minCount > 0 {
			policy = schedulingv1alpha3.CompositePodGroupSchedulingPolicy{
				Gang: &schedulingv1alpha3.CompositeGangSchedulingPolicy{MinGroupCount: minCount},
			}
		} else {
			policy = schedulingv1alpha3.CompositePodGroupSchedulingPolicy{
				Basic: &schedulingv1alpha3.CompositeBasicSchedulingPolicy{},
			}
		}
		cpg := &schedulingv1alpha3.CompositePodGroup{
			ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: ns},
			Spec: schedulingv1alpha3.CompositePodGroupSpec{
				SchedulingPolicy:            policy,
				ParentCompositePodGroupName: parentCPG,
			},
		}
		cache.AddCompositePodGroup(logger, cpg)
		return cpg
	}

	createPG := func(name string, minCount int32, parentCPG string) *schedulingv1beta1.PodGroup {
		var policy schedulingv1beta1.PodGroupSchedulingPolicy
		if minCount > 0 {
			policy = schedulingv1beta1.PodGroupSchedulingPolicy{
				Gang: &schedulingv1beta1.GangSchedulingPolicy{MinCount: minCount},
			}
		} else {
			policy = schedulingv1beta1.PodGroupSchedulingPolicy{
				Basic: &schedulingv1beta1.BasicSchedulingPolicy{},
			}
		}
		pg := &schedulingv1beta1.PodGroup{
			ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: ns},
			Spec: schedulingv1beta1.PodGroupSpec{
				SchedulingPolicy:            policy,
				ParentCompositePodGroupName: new(parentCPG),
			},
		}
		cache.AddPodGroup(pg)
		return pg
	}

	rootCPG := createCPG("cpg-root", 2, nil)

	pg1 := createPG("pg1", 3, "cpg-root")
	pg2 := createPG("pg2", 3, "cpg-root")
	pg3 := createPG("pg3", 3, "cpg-root")

	queuedPodInfos := make(map[fwk.EntityKey][]*framework.QueuedPodInfo)
	pgPods := make(map[string][]*v1.Pod)
	var allPods []*v1.Pod

	createPods := func(pgName string, count int, schedulable bool) {
		for i := range count {
			pod := st.MakePod().Namespace(ns).Name(fmt.Sprintf("%s-pod-%d", pgName, i)).UID(fmt.Sprintf("%s-pod-%d", pgName, i)).
				PodGroupName(pgName).Priority(100).Obj()

			if !schedulable && i == 0 {
				pod.Spec.Containers = []v1.Container{{
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("100")},
					},
				}}
			} else {
				pod.Spec.Containers = []v1.Container{{
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("1")},
					},
				}}
			}

			podInfo := &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: pod}}
			key := fwk.PodGroupKey(ns, pgName)
			queuedPodInfos[key] = append(queuedPodInfos[key], podInfo)
			pgPods[pgName] = append(pgPods[pgName], pod)
			allPods = append(allPods, pod)
			cache.AddPodGroupMember(pod)
		}
	}

	createPods("pg1", 3, true)
	createPods("pg2", 3, true)
	createPods("pg3", 3, false)

	snapshot := internalcache.NewEmptySnapshot()
	clientObjs := []runtime.Object{testNode, rootCPG, pg1, pg2, pg3}
	for _, pod := range allPods {
		clientObjs = append(clientObjs, pod)
	}
	client := clientsetfake.NewSimpleClientset(clientObjs...)
	informerFactory := informers.NewSharedInformerFactory(client, 0)

	fakePlugin := &fakePodGroupPlugin{
		filterStatus: map[string]*fwk.Status{
			"pg1-pod-0": nil,
			"pg1-pod-1": nil,
			"pg1-pod-2": nil,
			"pg2-pod-0": nil,
			"pg2-pod-1": nil,
			"pg2-pod-2": nil,
			"pg3-pod-0": fwk.NewStatus(fwk.Unschedulable, "insufficient cpu"),
			"pg3-pod-1": fwk.NewStatus(fwk.Unschedulable, "insufficient cpu"),
			"pg3-pod-2": fwk.NewStatus(fwk.Unschedulable, "insufficient cpu"),
		},
	}

	gangPluginFactory := func(ctx context.Context, obj runtime.Object, handle fwk.Handle) (fwk.Plugin, error) {
		return gangscheduling.New(ctx, obj, handle, feature.Features{EnableTopologyAwareWorkloadScheduling: true, EnableCompositePodGroup: true})
	}

	registry := []tf.RegisterPluginFunc{
		tf.RegisterFilterPlugin(fakePlugin.Name(), func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
			return fakePlugin, nil
		}),
		tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
		tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
		tf.RegisterPermitPlugin(gangscheduling.Name, gangPluginFactory),
		tf.RegisterPluginAsExtensions(gangscheduling.Name, gangPluginFactory, "PlacementFeasible"),
	}

	queue := internalqueue.NewSchedulingQueue(nil, informerFactory)
	schedFwk, err := tf.NewFramework(ctx, registry, "test-scheduler",
		frameworkruntime.WithInformerFactory(informerFactory),
		frameworkruntime.WithSnapshotSharedLister(snapshot),
		frameworkruntime.WithPodNominator(queue),
		frameworkruntime.WithPodActivator(queue),
		frameworkruntime.WithClientSet(client),
		frameworkruntime.WithEventRecorder(events.NewFakeRecorder(100)),
		frameworkruntime.WithWaitingPods(frameworkruntime.NewWaitingPodsMap()),
		frameworkruntime.WithPodsInPreBind(frameworkruntime.NewPodsInPreBindMap()),
		frameworkruntime.WithPodGroupManager(cache),
	)
	informerFactory.Start(ctx.Done())
	informerFactory.WaitForCacheSync(ctx.Done())
	if err != nil {
		t.Fatalf("Failed to create new framework: %v", err)
	}

	handledPods := make(map[string]*fwk.Status)
	var lock sync.Mutex

	sched := &Scheduler{
		Profiles:         profile.Map{"test-scheduler": schedFwk},
		SchedulingQueue:  internalqueue.NewTestQueue(ctx, nil),
		Cache:            cache,
		client:           client,
		nodeInfoSnapshot: snapshot,
		FailureHandler: func(ctx context.Context, fwk framework.Framework, p *framework.QueuedPodInfo, status *fwk.Status, ni *fwk.NominatingInfo, start time.Time) {
			lock.Lock()
			defer lock.Unlock()
			handledPods[p.Pod.Name] = status
		},
	}
	sched.SchedulePod = sched.schedulePod

	podGroupInfo := &framework.QueuedPodGroupInfo{
		QueuedPodInfos: queuedPodInfos,
		PodGroupInfo: &framework.PodGroupInfo{
			Namespace:         ns,
			Name:              "cpg-root",
			Type:              fwk.CompositePodGroupKeyType,
			UnscheduledPods:   allPods,
			CompositePodGroup: rootCPG,
			Children: []*framework.PodGroupInfo{
				{Namespace: ns, Name: "pg1", Type: fwk.PodGroupKeyType, PodGroup: pg1, UnscheduledPods: pgPods["pg1"]},
				{Namespace: ns, Name: "pg2", Type: fwk.PodGroupKeyType, PodGroup: pg2, UnscheduledPods: pgPods["pg2"]},
				{Namespace: ns, Name: "pg3", Type: fwk.PodGroupKeyType, PodGroup: pg3, UnscheduledPods: pgPods["pg3"]},
			},
		},
	}

	if err := cache.UpdateSnapshot(logger, snapshot); err != nil {
		t.Fatalf("Failed to update snapshot: %v", err)
	}
	sched.podGroupCycle(ctx, schedFwk, framework.NewCycleState(), podGroupInfo, time.Now())

	lock.Lock()
	defer lock.Unlock()

	successPods := []string{
		"pg1-pod-0", "pg1-pod-1", "pg1-pod-2",
		"pg2-pod-0", "pg2-pod-1", "pg2-pod-2",
	}

	for _, p := range successPods {
		if status, ok := handledPods[p]; ok {
			t.Errorf("Expected pod %s to be scheduled successfully, but got error: %v", p, status.AsError())
		}
	}

	failPods := []string{
		"pg3-pod-0", "pg3-pod-1", "pg3-pod-2",
	}

	for _, p := range failPods {
		if _, ok := handledPods[p]; !ok {
			t.Errorf("Expected pod %s to fail scheduling, but it was scheduled successfully", p)
		}
	}
}

// TestCPGBasicWithGangChildren_Internal verifies that a Basic CompositePodGroup allows its ready
// child Gang groups to schedule independently of each other.
//
// Tree structure:
//
//	   cpg-root (Basic)
//	  /            \
//	pg1            pg2
//	(S)            (F)
//
// (S) = Success
// (F) = Fail
func TestCPGBasicWithGangChildren_Internal(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.CompositePodGroup:               true,
		features.GenericWorkload:                 true,
		features.TopologyAwareWorkloadScheduling: true,
	})

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	logger := klog.FromContext(ctx)

	testNode := st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "20"}).Obj()
	cache := internalcache.New(ctx, nil, true, true /* CompositePodGroup */)
	cache.AddNode(logger, testNode)

	ns := "default"

	createCPG := func(name string, minCount int32, parentCPG *string) *schedulingv1alpha3.CompositePodGroup {
		var policy schedulingv1alpha3.CompositePodGroupSchedulingPolicy
		if minCount > 0 {
			policy = schedulingv1alpha3.CompositePodGroupSchedulingPolicy{
				Gang: &schedulingv1alpha3.CompositeGangSchedulingPolicy{MinGroupCount: minCount},
			}
		} else {
			policy = schedulingv1alpha3.CompositePodGroupSchedulingPolicy{
				Basic: &schedulingv1alpha3.CompositeBasicSchedulingPolicy{},
			}
		}
		cpg := &schedulingv1alpha3.CompositePodGroup{
			ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: ns},
			Spec: schedulingv1alpha3.CompositePodGroupSpec{
				SchedulingPolicy:            policy,
				ParentCompositePodGroupName: parentCPG,
			},
		}
		cache.AddCompositePodGroup(logger, cpg)
		return cpg
	}

	createPG := func(name string, minCount int32, parentCPG string) *schedulingv1beta1.PodGroup {
		var policy schedulingv1beta1.PodGroupSchedulingPolicy
		if minCount > 0 {
			policy = schedulingv1beta1.PodGroupSchedulingPolicy{
				Gang: &schedulingv1beta1.GangSchedulingPolicy{MinCount: minCount},
			}
		} else {
			policy = schedulingv1beta1.PodGroupSchedulingPolicy{
				Basic: &schedulingv1beta1.BasicSchedulingPolicy{},
			}
		}
		pg := &schedulingv1beta1.PodGroup{
			ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: ns},
			Spec: schedulingv1beta1.PodGroupSpec{
				SchedulingPolicy:            policy,
				ParentCompositePodGroupName: new(parentCPG),
			},
		}
		cache.AddPodGroup(pg)
		return pg
	}

	rootCPG := createCPG("cpg-root", 0, nil)

	pg1 := createPG("pg1", 3, "cpg-root")
	pg2 := createPG("pg2", 3, "cpg-root")

	queuedPodInfos := make(map[fwk.EntityKey][]*framework.QueuedPodInfo)
	pgPods := make(map[string][]*v1.Pod)
	var allPods []*v1.Pod

	createPods := func(pgName string, count int, schedulable bool) {
		for i := range count {
			pod := st.MakePod().Namespace(ns).Name(fmt.Sprintf("%s-pod-%d", pgName, i)).UID(fmt.Sprintf("%s-pod-%d", pgName, i)).
				PodGroupName(pgName).Priority(100).Obj()

			if !schedulable && i == 0 {
				pod.Spec.Containers = []v1.Container{{
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("100")},
					},
				}}
			} else {
				pod.Spec.Containers = []v1.Container{{
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("1")},
					},
				}}
			}

			podInfo := &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: pod}}
			key := fwk.PodGroupKey(ns, pgName)
			queuedPodInfos[key] = append(queuedPodInfos[key], podInfo)
			pgPods[pgName] = append(pgPods[pgName], pod)
			allPods = append(allPods, pod)
			cache.AddPodGroupMember(pod)
		}
	}

	createPods("pg1", 3, true)
	createPods("pg2", 3, false)

	snapshot := internalcache.NewEmptySnapshot()
	clientObjs := []runtime.Object{testNode, rootCPG, pg1, pg2}
	for _, pod := range allPods {
		clientObjs = append(clientObjs, pod)
	}
	client := clientsetfake.NewSimpleClientset(clientObjs...)
	informerFactory := informers.NewSharedInformerFactory(client, 0)

	fakePlugin := &fakePodGroupPlugin{
		filterStatus: map[string]*fwk.Status{
			"pg1-pod-0": nil,
			"pg1-pod-1": nil,
			"pg1-pod-2": nil,
			"pg2-pod-0": fwk.NewStatus(fwk.Unschedulable, "insufficient cpu"),
			"pg2-pod-1": fwk.NewStatus(fwk.Unschedulable, "insufficient cpu"),
			"pg2-pod-2": fwk.NewStatus(fwk.Unschedulable, "insufficient cpu"),
		},
	}

	gangPluginFactory := func(ctx context.Context, obj runtime.Object, handle fwk.Handle) (fwk.Plugin, error) {
		return gangscheduling.New(ctx, obj, handle, feature.Features{EnableTopologyAwareWorkloadScheduling: true, EnableCompositePodGroup: true})
	}

	registry := []tf.RegisterPluginFunc{
		tf.RegisterFilterPlugin(fakePlugin.Name(), func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
			return fakePlugin, nil
		}),
		tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
		tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
		tf.RegisterPermitPlugin(gangscheduling.Name, gangPluginFactory),
		tf.RegisterPluginAsExtensions(gangscheduling.Name, gangPluginFactory, "PlacementFeasible"),
	}

	queue := internalqueue.NewSchedulingQueue(nil, informerFactory)
	schedFwk, err := tf.NewFramework(ctx, registry, "test-scheduler",
		frameworkruntime.WithInformerFactory(informerFactory),
		frameworkruntime.WithSnapshotSharedLister(snapshot),
		frameworkruntime.WithPodNominator(queue),
		frameworkruntime.WithPodActivator(queue),
		frameworkruntime.WithClientSet(client),
		frameworkruntime.WithEventRecorder(events.NewFakeRecorder(100)),
		frameworkruntime.WithWaitingPods(frameworkruntime.NewWaitingPodsMap()),
		frameworkruntime.WithPodsInPreBind(frameworkruntime.NewPodsInPreBindMap()),
		frameworkruntime.WithPodGroupManager(cache),
	)
	informerFactory.Start(ctx.Done())
	informerFactory.WaitForCacheSync(ctx.Done())
	if err != nil {
		t.Fatalf("Failed to create new framework: %v", err)
	}

	handledPods := make(map[string]*fwk.Status)
	var lock sync.Mutex

	sched := &Scheduler{
		Profiles:         profile.Map{"test-scheduler": schedFwk},
		SchedulingQueue:  internalqueue.NewTestQueue(ctx, nil),
		Cache:            cache,
		client:           client,
		nodeInfoSnapshot: snapshot,
		FailureHandler: func(ctx context.Context, fwk framework.Framework, p *framework.QueuedPodInfo, status *fwk.Status, ni *fwk.NominatingInfo, start time.Time) {
			lock.Lock()
			defer lock.Unlock()
			handledPods[p.Pod.Name] = status
		},
	}
	sched.SchedulePod = sched.schedulePod

	podGroupInfo := &framework.QueuedPodGroupInfo{
		QueuedPodInfos: queuedPodInfos,
		PodGroupInfo: &framework.PodGroupInfo{
			Namespace:         ns,
			Name:              "cpg-root",
			Type:              fwk.CompositePodGroupKeyType,
			UnscheduledPods:   allPods,
			CompositePodGroup: rootCPG,
			Children: []*framework.PodGroupInfo{
				{Namespace: ns, Name: "pg1", Type: fwk.PodGroupKeyType, PodGroup: pg1, UnscheduledPods: pgPods["pg1"]},
				{Namespace: ns, Name: "pg2", Type: fwk.PodGroupKeyType, PodGroup: pg2, UnscheduledPods: pgPods["pg2"]},
			},
		},
	}

	if err := cache.UpdateSnapshot(logger, snapshot); err != nil {
		t.Fatalf("Failed to update snapshot: %v", err)
	}
	sched.podGroupCycle(ctx, schedFwk, framework.NewCycleState(), podGroupInfo, time.Now())

	lock.Lock()
	defer lock.Unlock()

	successPods := []string{
		"pg1-pod-0", "pg1-pod-1", "pg1-pod-2",
	}

	for _, p := range successPods {
		if status, ok := handledPods[p]; ok {
			t.Errorf("Expected pod %s to be scheduled successfully, but got error: %v", p, status.AsError())
		}
	}

	failPods := []string{
		"pg2-pod-0", "pg2-pod-1", "pg2-pod-2",
	}

	for _, p := range failPods {
		if _, ok := handledPods[p]; !ok {
			t.Errorf("Expected pod %s to fail scheduling, but it was scheduled successfully", p)
		}
	}
}

// fakeAssignmentRecordingPlugin captures pod group assignments during ScorePlacement
// to verify that only feasible pods from successful branches of the hierarchy are scored.
type fakeAssignmentRecordingPlugin struct {
	mu          sync.Mutex
	name        string
	assignments map[string]*fwk.PodGroupAssignments
}

var _ fwk.PlacementScorePlugin = &fakeAssignmentRecordingPlugin{}

func (p *fakeAssignmentRecordingPlugin) Name() string {
	return p.name
}

func (p *fakeAssignmentRecordingPlugin) ScorePlacement(ctx context.Context, state fwk.PlacementCycleState, podGroup fwk.PodGroupInfo, placement *fwk.PodGroupAssignments) (int64, *fwk.Status) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.assignments == nil {
		p.assignments = make(map[string]*fwk.PodGroupAssignments)
	}
	p.assignments[placement.Placement.Name] = placement
	return 1, nil
}

func (p *fakeAssignmentRecordingPlugin) PlacementScoreExtensions() fwk.PlacementScoreExtensions {
	return nil
}

func TestScorePlacementPodGroupAssignments(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.TopologyAwareWorkloadScheduling: true,
		features.GenericWorkload:                 true,
		features.CompositePodGroup:               true,
	})

	type tree struct {
		name     string
		children []tree
	}

	makePodRes := func(podName, nodeName string, status *fwk.Status) algorithmResult {
		p := st.MakePod().Name(podName).UID(podName).Obj()
		podInfo, _ := framework.NewPodInfo(p)
		return algorithmResult{
			podInfo:        &framework.QueuedPodInfo{PodInfo: podInfo},
			status:         status,
			scheduleResult: ScheduleResult{SuggestedHost: nodeName},
		}
	}

	tests := map[string]struct {
		tree                tree
		results             map[string]map[string]*podGroupAlgorithmResult
		expectedAssignments map[string]map[string]string
	}{
		"flat composite pod group with all successful leaves": {
			tree: tree{
				name: "rootcpg",
				children: []tree{
					{name: "pg1"},
					{name: "pg2"},
				},
			},
			results: map[string]map[string]*podGroupAlgorithmResult{
				"placement1": {
					"rootcpg": {
						status: fwk.NewStatus(fwk.Success),
					},
					"pg1": {
						status: fwk.NewStatus(fwk.Success),
						podResults: []algorithmResult{
							makePodRes("p1", "node1", fwk.NewStatus(fwk.Success)),
						},
					},
					"pg2": {
						status: fwk.NewStatus(fwk.Success),
						podResults: []algorithmResult{
							makePodRes("p2", "node2", fwk.NewStatus(fwk.Success)),
						},
					},
				},
				"placement2": {
					"rootcpg": {
						status: fwk.NewStatus(fwk.Success),
					},
					"pg1": {
						status: fwk.NewStatus(fwk.Success),
						podResults: []algorithmResult{
							makePodRes("p1", "node3", fwk.NewStatus(fwk.Success)),
						},
					},
					"pg2": {
						status: fwk.NewStatus(fwk.Success),
						podResults: []algorithmResult{
							makePodRes("p2", "node4", fwk.NewStatus(fwk.Success)),
						},
					},
				},
			},
			expectedAssignments: map[string]map[string]string{
				"placement1": {"p1": "node1", "p2": "node2"},
				"placement2": {"p1": "node3", "p2": "node4"},
			},
		},
		"multi-level tree where an intermediate subtree fails": {
			tree: tree{
				name: "rootcpg",
				children: []tree{
					{
						name:     "midcpg1",
						children: []tree{{name: "pg1"}},
					},
					{
						name:     "midcpg2",
						children: []tree{{name: "pg2"}},
					},
				},
			},
			results: map[string]map[string]*podGroupAlgorithmResult{
				"placement1": {
					"rootcpg": {
						status: fwk.NewStatus(fwk.Success),
					},
					"midcpg1": {
						status: fwk.NewStatus(fwk.Success),
					},
					"pg1": {
						status: fwk.NewStatus(fwk.Success),
						podResults: []algorithmResult{
							makePodRes("p1", "node1", fwk.NewStatus(fwk.Success)),
						},
					},
					"midcpg2": {
						status: fwk.NewStatus(fwk.Unschedulable, "minGroupCount not met"),
					},
					"pg2": {
						status: fwk.NewStatus(fwk.Success),
						podResults: []algorithmResult{
							makePodRes("p2", "node2", fwk.NewStatus(fwk.Success)),
						},
					},
				},
				"placement2": {
					"rootcpg": {
						status: fwk.NewStatus(fwk.Success),
					},
					"midcpg1": {
						status: fwk.NewStatus(fwk.Success),
					},
					"pg1": {
						status: fwk.NewStatus(fwk.Success),
						podResults: []algorithmResult{
							makePodRes("p1", "node3", fwk.NewStatus(fwk.Success)),
						},
					},
					"midcpg2": {
						status: fwk.NewStatus(fwk.Success),
					},
					"pg2": {
						status: fwk.NewStatus(fwk.Success),
						podResults: []algorithmResult{
							makePodRes("p2", "node4", fwk.NewStatus(fwk.Success)),
						},
					},
				},
			},
			expectedAssignments: map[string]map[string]string{
				"placement1": {"p1": "node1"},
				"placement2": {"p1": "node3", "p2": "node4"},
			},
		},
		"missing subtree result due to short-circuiting in PlacementFeasible": {
			tree: tree{
				name: "rootcpg",
				children: []tree{
					{
						name:     "midcpg1",
						children: []tree{{name: "pg1"}},
					},
					{
						name:     "midcpg2",
						children: []tree{{name: "pg2"}},
					},
				},
			},
			results: map[string]map[string]*podGroupAlgorithmResult{
				"placement1": {
					"rootcpg": {
						status: fwk.NewStatus(fwk.Success),
					},
					"midcpg1": {
						status: fwk.NewStatus(fwk.Success),
					},
					"pg1": {
						status: fwk.NewStatus(fwk.Success),
						podResults: []algorithmResult{
							makePodRes("p1", "node1", fwk.NewStatus(fwk.Success)),
						},
					},
				},
				"placement2": {
					"rootcpg": {
						status: fwk.NewStatus(fwk.Success),
					},
					"midcpg2": {
						status: fwk.NewStatus(fwk.Success),
					},
					"pg2": {
						status: fwk.NewStatus(fwk.Success),
						podResults: []algorithmResult{
							makePodRes("p2", "node2", fwk.NewStatus(fwk.Success)),
						},
					},
				},
			},
			expectedAssignments: map[string]map[string]string{
				"placement1": {"p1": "node1"},
				"placement2": {"p2": "node2"},
			},
		},
		"leaf pod group with some failed or unassigned pods": {
			tree: tree{
				name: "rootcpg",
				children: []tree{
					{name: "pg1"},
					{name: "pg3"},
				},
			},
			results: map[string]map[string]*podGroupAlgorithmResult{
				"placement1": {
					"rootcpg": {
						status: fwk.NewStatus(fwk.Success),
					},
					"pg1": {
						status: fwk.NewStatus(fwk.Success),
						podResults: []algorithmResult{
							makePodRes("p1", "node1", fwk.NewStatus(fwk.Success)),
						},
					},
					"pg3": {
						status: fwk.NewStatus(fwk.Success),
						podResults: []algorithmResult{
							makePodRes("p3", "node3", fwk.NewStatus(fwk.Success)),
							makePodRes("p4", "", fwk.NewStatus(fwk.Unschedulable)),
						},
					},
				},
				"placement2": {
					"rootcpg": {
						status: fwk.NewStatus(fwk.Success),
					},
					"pg1": {
						status: fwk.NewStatus(fwk.Success),
						podResults: []algorithmResult{
							makePodRes("p1", "node2", fwk.NewStatus(fwk.Success)),
						},
					},
					"pg3": {
						status: fwk.NewStatus(fwk.Success),
						podResults: []algorithmResult{
							makePodRes("p3", "", fwk.NewStatus(fwk.Success)),
							makePodRes("p4", "node4", fwk.NewStatus(fwk.Success)),
						},
					},
				},
			},
			expectedAssignments: map[string]map[string]string{
				"placement1": {"p1": "node1", "p3": "node3"},
				"placement2": {"p1": "node2", "p4": "node4"},
			},
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)

			informerFactory := informers.NewSharedInformerFactory(clientsetfake.NewClientset(), 0)
			queue := internalqueue.NewSchedulingQueue(nil, informerFactory)

			plugin := &fakeAssignmentRecordingPlugin{name: "fake-assignment-recorder"}
			registry := []tf.RegisterPluginFunc{
				tf.RegisterPlacementScorePlugin(plugin.Name(), func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
					return plugin, nil
				}, 1),
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

			nameToKey := make(map[string]fwk.EntityKey)
			var buildTree func(node tree) *framework.PodGroupInfo
			buildTree = func(node tree) *framework.PodGroupInfo {
				var children []*framework.PodGroupInfo
				for _, ch := range node.children {
					children = append(children, buildTree(ch))
				}
				pgi := &framework.PodGroupInfo{
					Name:      node.name,
					Namespace: "default",
					Children:  children,
				}
				if len(children) > 0 {
					pgi.Type = fwk.CompositePodGroupKeyType
					pgi.CompositePodGroup = st.MakeCompositePodGroup().Name(node.name).Namespace("default").Obj()
				} else {
					pgi.Type = fwk.PodGroupKeyType
					pgi.PodGroup = st.MakePodGroup().Name(node.name).Namespace("default").Obj()
				}
				nameToKey[node.name] = pgKey(pgi)
				return pgi
			}

			root := buildTree(tc.tree)

			sched := &Scheduler{}
			successfulResults := make(map[*fwk.Placement]map[fwk.EntityKey]*podGroupAlgorithmResult)
			for placementName, resMap := range tc.results {
				placement := &fwk.Placement{Name: placementName}
				entityMap := make(map[fwk.EntityKey]*podGroupAlgorithmResult, len(resMap))
				for groupName, res := range resMap {
					key, ok := nameToKey[groupName]
					if !ok {
						t.Fatalf("Unknown pod group name in results: %s", groupName)
					}
					if groupName == tc.tree.name && res.placementCycleState == nil {
						res.placementCycleState = framework.NewCycleState()
					}
					entityMap[key] = res
				}
				successfulResults[placement] = entityMap
			}

			_, status := sched.findBestCompositePodGroupPlacement(ctx, schedFwk, framework.NewCycleState(), root, successfulResults)
			if !status.IsSuccess() {
				t.Fatalf("Expected findBestCompositePodGroupPlacement to succeed, got status: %v", status)
			}

			gotAssignments := make(map[string]map[string]string)
			for placementName, pga := range plugin.assignments {
				pods := make(map[string]string)
				for _, pa := range pga.ProposedAssignments {
					pods[pa.GetPod().Name] = pa.GetNodeName()
				}
				gotAssignments[placementName] = pods
			}
			if diff := cmp.Diff(tc.expectedAssignments, gotAssignments); diff != "" {
				t.Errorf("Unexpected pod group assignments in ScorePlacement (-want,+got):\n%s", diff)
			}
		})
	}
}
