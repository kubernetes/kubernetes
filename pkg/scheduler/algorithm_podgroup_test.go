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
	"maps"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"

	v1 "k8s.io/api/core/v1"
	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/tools/events"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	componentmetrics "k8s.io/component-base/metrics"
	"k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	internalcache "k8s.io/kubernetes/pkg/scheduler/backend/cache"
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
	"k8s.io/utils/ptr"
)

func testPodGroupAlgorithm(t *testing.T, ctx context.Context, snapshot *internalcache.Snapshot, cache internalcache.Cache, queue internalqueue.SchedulingQueue) *PodGroupSchedulingAlgorithm {
	t.Helper()
	if cache == nil {
		cache = internalcache.New(ctx, nil, true, false)
	}
	alg, err := NewSchedulingAlgorithm(snapshot, cache, WithCurrentCycleProvider(queue.SchedulingCycle))
	if err != nil {
		t.Fatalf("Failed to create scheduling algorithm: %v", err)
	}
	return NewPodGroupAlgorithm(snapshot, alg)
}

func TestPodGroupSchedulingAlgorithm(t *testing.T) {
	testNode := st.MakeNode().Name("node1").UID("node1").Obj()

	p1 := st.MakePod().Namespace("default").Name("p1").UID("p1").PodGroupName("pg").SchedulerName("test-scheduler").Obj()
	p2 := st.MakePod().Namespace("default").Name("p2").UID("p2").PodGroupName("pg").SchedulerName("test-scheduler").Obj()
	p3 := st.MakePod().Namespace("default").Name("p3").UID("p3").PodGroupName("pg").SchedulerName("test-scheduler").Obj()

	testPodGroup := &schedulingv1beta1.PodGroup{
		ObjectMeta: metav1.ObjectMeta{Name: "pg", Namespace: "default"},
	}

	qInfo1 := &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: p1}}
	qInfo2 := &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: p2}}
	qInfo3 := &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: p3}}
	queuedPodInfos := []*framework.QueuedPodInfo{qInfo1, qInfo2, qInfo3}

	podGroupInfo := newQueuedPodGroupInfo(&framework.PodGroupInfo{GenericPodGroup: fwk.NewGenericPodGroup(testPodGroup)}, queuedPodInfos...)

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
						tf.RegisterPlacementFeasiblePlugin(placementFeasiblePlugin.Name(), func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
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
					cache.AddGenericPodGroup(fwk.NewGenericPodGroup(testPodGroup))

					if err := cache.UpdateSnapshot(logger, snapshot); err != nil {
						t.Fatalf("Failed to update snapshot: %v", err)
					}

					podGroupAlgorithm := testPodGroupAlgorithm(t, ctx, snapshot, cache, queue)
					resultsMap := podGroupAlgorithm.RunRootSchedulingAlgorithm(ctx, schedFwk, framework.NewCycleState(), podGroupInfo)
					result := resultsMap[podGroupInfo.PodGroupInfo.GetKey()]

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
	podGroupPod := st.MakePod().Namespace("default").Name("foo").UID("foo").PodGroupName("pg").Obj()
	testPodGroup := &schedulingv1beta1.PodGroup{
		ObjectMeta: metav1.ObjectMeta{Name: "pg", Namespace: "default"},
	}

	podInfo, err := framework.NewPodInfo(st.MakePod().Namespace("default").Name("foo").UID("foo").PodGroupName("pg").Obj())
	if err != nil {
		t.Fatalf("Failed to create pod info: %v", err)
	}
	podGroupPodInfo := &framework.QueuedPodInfo{PodInfo: podInfo}

	pgKeyVal := fwk.PodGroupKey("default", "pg")
	queuedPodInfos := []*framework.QueuedPodInfo{{PodInfo: &framework.PodInfo{Pod: podGroupPod}}}
	pgInfo := newQueuedPodGroupInfo(&framework.PodGroupInfo{GenericPodGroup: fwk.NewGenericPodGroup(testPodGroup)}, queuedPodInfos...)

	tests := map[string]struct {
		placementPlugin               fakePlacementPlugin
		placementFeasibleStatuses     [][]fwk.Code
		expectedResult                PodGroupScheduledResult
		expectedGeneratedPlacements   int
		expectedFeasibleEvaluations   int
		expectedInfeasibleEvaluations int
	}{
		"respects higher score of placement1": {
			expectedGeneratedPlacements: 2,
			expectedFeasibleEvaluations: 2,
			placementPlugin: fakePlacementPlugin{
				generatePlacementsResult: map[fwk.EntityKey]map[string][]string{
					pgKeyVal: {
						"placement1": {nodes[0].Name},
						"placement2": {nodes[1].Name},
					},
				},
				scorePlacementsResult: map[fwk.EntityKey]map[string]int64{
					pgKeyVal: {
						"placement1": 2,
						"placement2": 1,
					},
				},
			},
			expectedResult: PodGroupScheduledResult{
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
				generatePlacementsResult: map[fwk.EntityKey]map[string][]string{
					pgKeyVal: {
						"placement1": {nodes[0].Name},
						"placement2": {nodes[1].Name},
					},
				},
				scorePlacementsResult: map[fwk.EntityKey]map[string]int64{
					pgKeyVal: {
						"placement1": 1,
						"placement2": 2,
					},
				},
			},
			expectedResult: PodGroupScheduledResult{
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
				generatePlacementsResult: map[fwk.EntityKey]map[string][]string{},
			},
			expectedResult: PodGroupScheduledResult{
				status: fwk.NewStatus(fwk.Unschedulable, "no feasible placements found").WithPlugin("FakePlacementPlugin_Ordered"),
			},
		},
		"when all placements are infeasible, returns unschedulable": {
			expectedGeneratedPlacements:   2,
			expectedInfeasibleEvaluations: 2,
			placementPlugin: fakePlacementPlugin{
				generatePlacementsResult: map[fwk.EntityKey]map[string][]string{
					pgKeyVal: {
						"placement1": {nodes[0].Name},
						"placement2": {nodes[1].Name},
					},
				},
				scorePlacementsResult: map[fwk.EntityKey]map[string]int64{
					pgKeyVal: {
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
			expectedResult: PodGroupScheduledResult{
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
				generatePlacementsResult: map[fwk.EntityKey]map[string][]string{
					pgKeyVal: {
						"placement1": {nodes[0].Name},
						"placement2": {nodes[1].Name},
					},
				},
				scorePlacementsResult: map[fwk.EntityKey]map[string]int64{
					pgKeyVal: {
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
			expectedResult: PodGroupScheduledResult{
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
				generatePlacementsResult: map[fwk.EntityKey]map[string][]string{
					pgKeyVal: {
						"placement1": {nodes[0].Name},
						"placement2": {nodes[1].Name},
					},
				},
				scorePlacementsResult: map[fwk.EntityKey]map[string]int64{
					pgKeyVal: {
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
			expectedResult: PodGroupScheduledResult{
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
				generatePlacementsResult: map[fwk.EntityKey]map[string][]string{
					pgKeyVal: {
						"placement1": {nodes[0].Name},
						"placement2": {nodes[1].Name},
					},
				},
				scorePlacementsResult: map[fwk.EntityKey]map[string]int64{
					pgKeyVal: {
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
			expectedResult: PodGroupScheduledResult{
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
				generatePlacementsStatus: map[fwk.EntityKey]*fwk.Status{pgKeyVal: fwk.NewStatus(fwk.Error, "error for test")},
			},
			expectedResult: PodGroupScheduledResult{
				status: fwk.NewStatus(fwk.Error, "error for test").WithPlugin("FakePlacementPlugin"),
			},
		},
		"when score plugin fails, returns error": {
			expectedGeneratedPlacements: 2,
			expectedFeasibleEvaluations: 2,
			placementPlugin: fakePlacementPlugin{
				generatePlacementsResult: map[fwk.EntityKey]map[string][]string{
					pgKeyVal: {
						"placement1": {nodes[0].Name},
						"placement2": {nodes[1].Name},
					},
				},
				scorePlacementsResult: map[fwk.EntityKey]map[string]int64{
					pgKeyVal: {
						"placement1": 1,
					},
				},
				scorePlacementsStatus: map[fwk.EntityKey]map[string]*fwk.Status{
					pgKeyVal: {
						"placement2": fwk.NewStatus(fwk.Error, "error for test"),
					},
				},
			},
			expectedResult: PodGroupScheduledResult{
				status: fwk.NewStatus(fwk.Error, `running PlacementScore plugins: plugin "FakePlacementPlugin" failed with: error for test`).WithPlugin("FakePlacementPlugin"),
			},
		},
		"when a placement evaluation errors, returns error": {
			expectedGeneratedPlacements: 1,
			placementPlugin: fakePlacementPlugin{
				generatePlacementsResult: map[fwk.EntityKey]map[string][]string{
					pgKeyVal: {
						"placement1": {nodes[0].Name},
					},
				},
				scorePlacementsResult: map[fwk.EntityKey]map[string]int64{
					pgKeyVal: {
						"placement1": 1,
					},
				},
				filterStatus: map[string]*fwk.Status{
					nodes[0].Name: fwk.NewStatus(fwk.Error, "error for test"),
				},
			},
			expectedResult: PodGroupScheduledResult{
				podResults: []algorithmResult{
					{
						podInfo: podGroupPodInfo,
						scheduleResult: ScheduleResult{
							nominatingInfo: &fwk.NominatingInfo{NominatingMode: fwk.ModeOverride},
						},
						status: fwk.NewStatus(fwk.Error, `running "FakePlacementPlugin" filter plugin: error for test`),
					},
				},
				status: fwk.NewStatus(fwk.Error, `failed to schedule other pod from a pod group: running "FakePlacementPlugin" filter plugin: error for test`),
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
					tf.RegisterPlacementFeasiblePlugin(placementFeasiblePlugin.Name(), func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
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
				cache.AddGenericPodGroup(fwk.NewGenericPodGroup(testPodGroup))

				if err := cache.UpdateSnapshot(logger, snapshot); err != nil {
					t.Fatalf("Failed to update snapshot: %v", err)
				}

				metrics.GeneratedPlacementsTotal.Reset()
				metrics.PlacementEvaluations.Reset()
				metrics.PlacementEvaluationDuration.Reset()

				podGroupAlgorithm := testPodGroupAlgorithm(t, ctx, snapshot, cache, queue)
				resultsMap := podGroupAlgorithm.RunRootSchedulingAlgorithm(ctx, schedFwk, framework.NewCycleState(), pgInfo)
				result := resultsMap[pgInfo.PodGroupInfo.GetKey()]

				if result.podGroupInfo != pgInfo.PodGroupInfo {
					t.Errorf("Unexpected podGroupInfo field (-want,+got):\n- %v\n+ %v", pgInfo, result.podGroupInfo)
				}

				opts := cmp.Options{
					cmp.AllowUnexported(
						PodGroupScheduledResult{},
						algorithmResult{},
						ScheduleResult{},
						fwk.Status{},
						framework.PodInfo{}),
					cmpopts.IgnoreFields(PodGroupScheduledResult{}, "podGroupInfo", "placementCycleState", "anyScheduled"),
					cmpopts.IgnoreFields(algorithmResult{}, "podCtx", "schedulingDuration"),
					statusCmpOpt,
				}

				if diff := cmp.Diff(tt.expectedResult, *result, opts...); diff != "" {
					t.Fatalf("Unexpected algorithm result (-want,+got):\n%s", diff)
				}

				baseLabels := map[string]string{"profile": "test-scheduler"}
				fixedLabels := map[string]string{"profile": "test-scheduler", "entity": metrics.PodGroup}
				feasibleLabels := map[string]string{"profile": "test-scheduler", "result": metrics.FeasibleResult, "entity": metrics.PodGroup}
				infeasibleLabels := map[string]string{"profile": "test-scheduler", "result": metrics.InfeasibleResult, "entity": metrics.PodGroup}
				assertCounterValueFromGatherer(t, testRegistry, "scheduler_generated_placements_total", baseLabels, "entity", metrics.PodGroup, tt.expectedGeneratedPlacements)
				assertCounterValueFromGatherer(t, testRegistry, "scheduler_placement_evaluations_total", fixedLabels, "result", metrics.FeasibleResult, tt.expectedFeasibleEvaluations)
				assertCounterValueFromGatherer(t, testRegistry, "scheduler_placement_evaluations_total", fixedLabels, "result", metrics.InfeasibleResult, tt.expectedInfeasibleEvaluations)
				assertHistogramSampleCountFromGatherer(t, testRegistry, "scheduler_placement_evaluation_duration_seconds", feasibleLabels, tt.expectedFeasibleEvaluations)
				assertHistogramSampleCountFromGatherer(t, testRegistry, "scheduler_placement_evaluation_duration_seconds", infeasibleLabels, tt.expectedInfeasibleEvaluations)

				// Verify compositepodgroup metrics are 0
				cpgFixedLabels := map[string]string{"profile": "test-scheduler", "entity": metrics.CompositePodGroup}
				assertCounterValueFromGatherer(t, testRegistry, "scheduler_generated_placements_total", baseLabels, "entity", metrics.CompositePodGroup, 0)
				assertCounterValueFromGatherer(t, testRegistry, "scheduler_placement_evaluations_total", cpgFixedLabels, "result", metrics.FeasibleResult, 0)
				assertCounterValueFromGatherer(t, testRegistry, "scheduler_placement_evaluations_total", cpgFixedLabels, "result", metrics.InfeasibleResult, 0)
			})
		}
	}
}

func placementWithNodes(name string, nodeNames ...string) *fwk.Placement {
	nodes := make([]fwk.NodeInfo, 0, len(nodeNames))
	for _, n := range nodeNames {
		ni := framework.NewNodeInfo()
		ni.SetNode(&v1.Node{Name: n})
		nodes = append(nodes, ni)
	}
	return &fwk.Placement{Name: name, Nodes: nodes}
}

func podGroupWithNominations(nominated ...string) *framework.QueuedPodGroupInfo {
	infos := make([]*framework.QueuedPodInfo, 0, len(nominated))
	for i, nnn := range nominated {
		pod := st.MakePod().Name(string(rune('a' + i))).Namespace("default").PodGroupName("pg").NominatedNodeName(nnn).Obj()
		infos = append(infos, &framework.QueuedPodInfo{
			PodInfo: &framework.PodInfo{Pod: pod},
		})
	}
	return newQueuedPodGroupInfo(&framework.PodGroupInfo{
		GenericPodGroup: fwk.NewGenericPodGroup(st.MakePodGroup().Name("pg").Namespace("default").Obj()),
	}, infos...)
}

func TestNominatedPlacement(t *testing.T) {
	rack1 := placementWithNodes("rack-1", "node-1")
	rack2 := placementWithNodes("rack-2", "node-2")
	rack12 := placementWithNodes("rack-12", "node-1", "node-2")
	rack23 := placementWithNodes("rack-23", "node-2", "node-3")

	tests := []struct {
		name       string
		placements []*fwk.Placement
		podGroup   *framework.QueuedPodGroupInfo
		want       *fwk.Placement
	}{
		{
			name:       "no nominations returns nil",
			placements: []*fwk.Placement{rack1, rack2},
			podGroup:   podGroupWithNominations("", ""),
			want:       nil,
		},
		{
			name:       "nominated node not in any placement returns nil",
			placements: []*fwk.Placement{rack1, rack2},
			podGroup:   podGroupWithNominations("node-3"),
			want:       nil,
		},
		{
			name:       "single placement holds the nominated node",
			placements: []*fwk.Placement{rack1, rack2},
			podGroup:   podGroupWithNominations("node-2", ""),
			want:       rack2,
		},
		{
			name:       "single placement matching several nominated nodes wins",
			placements: []*fwk.Placement{rack1, rack23},
			podGroup:   podGroupWithNominations("node-2", "node-3"),
			want:       rack23,
		},
		{
			name:       "overlapping placements sharing a nominated node returns nil",
			placements: []*fwk.Placement{rack12, rack23},
			podGroup:   podGroupWithNominations("node-2"),
			want:       nil,
		},
		{
			name:       "distinct placements each holding a nominated node returns nil",
			placements: []*fwk.Placement{rack1, rack2},
			podGroup:   podGroupWithNominations("node-1", "node-2"),
			want:       nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := nominatedPlacement(tt.placements, tt.podGroup.PodGroupInfo, tt.podGroup); got != tt.want {
				t.Errorf("nominatedPlacement() = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestNominatedPlacementOrderIndependent asserts that when nominated nodes fall into overlapping
// placements the fallback (nil) is returned regardless of the order the generator returned them,
// so placement selection can't depend on slice order.
func TestNominatedPlacementOrderIndependent(t *testing.T) {
	rack12 := placementWithNodes("rack-12", "node-1", "node-2")
	rack23 := placementWithNodes("rack-23", "node-2", "node-3")
	pg := podGroupWithNominations("node-2")

	forward := nominatedPlacement([]*fwk.Placement{rack12, rack23}, pg.PodGroupInfo, pg)
	reverse := nominatedPlacement([]*fwk.Placement{rack23, rack12}, pg.PodGroupInfo, pg)
	if forward != nil || reverse != nil {
		t.Errorf("nominatedPlacement() = %v (forward), %v (reverse), want nil for both", forward, reverse)
	}
}

// TestPodGroupSchedulingPlacementAlgorithm_NominatedNode covers the NominatedNodeName path:
// the placement matching the pods' NNN is evaluated first, short-circuits scoring when the
// gang is feasible there, and otherwise falls through to normal score-based selection.
func TestPodGroupSchedulingPlacementAlgorithm_NominatedNode(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.TopologyAwareWorkloadScheduling: true,
		features.GenericWorkload:                 true,
	})

	nodes := []*v1.Node{
		st.MakeNode().Name("node1").Obj(),
		st.MakeNode().Name("node2").Obj(),
	}

	podGroupKey := fwk.PodGroupKey("default", "pg")
	tests := map[string]struct {
		placementPlugin           fakePlacementPlugin
		placementFeasibleStatuses [][]fwk.Code
		// nominatedNodeName is written to the pod's status before scheduling.
		nominatedNodeName string
		// rootIsCPG makes the scheduling root a CompositePodGroup, which gates off the NNN fast path.
		rootIsCPG bool
		// expectedHost is the node the gang should land on. Empty means unschedulable.
		expectedHost   string
		expectedStatus *fwk.Status
		expectError    bool
	}{
		"nominated placement short-circuits scoring even with a lower score": {
			placementPlugin: fakePlacementPlugin{
				generatePlacementsResult: map[fwk.EntityKey]map[string][]string{
					podGroupKey: {
						"placement1": {nodes[0].Name},
						"placement2": {nodes[1].Name},
					},
				},
				// placement1 scores higher, so without the NNN short-circuit the gang would
				// land on node1. The nomination points at node2, which must win instead.
				scorePlacementsResult: map[fwk.EntityKey]map[string]int64{
					podGroupKey: {
						"placement1": 2,
						"placement2": 1,
					},
				},
			},
			nominatedNodeName: nodes[1].Name,
			expectedHost:      nodes[1].Name,
		},
		"infeasible nominated placement falls back to scoring": {
			placementPlugin: fakePlacementPlugin{
				generatePlacementsResult: map[fwk.EntityKey]map[string][]string{
					podGroupKey: {
						"placement1": {nodes[0].Name},
						"placement2": {nodes[1].Name},
					},
				},
				scorePlacementsResult: map[fwk.EntityKey]map[string]int64{
					podGroupKey: {
						"placement1": 1,
					},
				},
				// The nominated node is infeasible, so its placement can't short-circuit and
				// the gang falls back to the only feasible placement.
				filterStatus: map[string]*fwk.Status{
					nodes[1].Name: fwk.NewStatus(fwk.Unschedulable),
				},
			},
			placementFeasibleStatuses: [][]fwk.Code{{fwk.Unschedulable}, {fwk.Success}},
			nominatedNodeName:         nodes[1].Name,
			expectedHost:              nodes[0].Name,
		},
		"error on the nominated placement propagates": {
			placementPlugin: fakePlacementPlugin{
				generatePlacementsResult: map[fwk.EntityKey]map[string][]string{
					podGroupKey: {
						"placement1": {nodes[0].Name},
						"placement2": {nodes[1].Name},
					},
				},
				scorePlacementsResult: map[fwk.EntityKey]map[string]int64{
					podGroupKey: {
						"placement1": 1,
					},
				},
				filterStatus: map[string]*fwk.Status{
					nodes[1].Name: fwk.NewStatus(fwk.Error, "injected filter error"),
				},
			},
			nominatedNodeName: nodes[1].Name,
			expectError:       true,
		},
		"CPG root skips the nominated fast path and scores every placement": {
			placementPlugin: fakePlacementPlugin{
				generatePlacementsResult: map[fwk.EntityKey]map[string][]string{
					podGroupKey: {
						"placement1": {nodes[0].Name},
						"placement2": {nodes[1].Name},
					},
				},
				// Same setup as the short-circuit case: placement1 scores higher and the
				// nomination points at node2. Because the root is a CompositePodGroup, the NNN
				// fast path is disabled and the higher-scored placement1 (node1) must win.
				scorePlacementsResult: map[fwk.EntityKey]map[string]int64{
					podGroupKey: {
						"placement1": 2,
						"placement2": 1,
					},
				},
			},
			nominatedNodeName: nodes[1].Name,
			rootIsCPG:         true,
			expectedHost:      nodes[0].Name,
		},
		"feasible nominated placement that schedules no new pod does not short-circuit": {
			placementPlugin: fakePlacementPlugin{
				generatePlacementsResult: map[fwk.EntityKey]map[string][]string{
					podGroupKey: {
						"placement1": {nodes[0].Name},
						"placement2": {nodes[1].Name},
					},
				},
				// The nominated placement2 even scores higher, so nothing but the anyScheduled
				// guard keeps it from winning.
				scorePlacementsResult: map[fwk.EntityKey]map[string]int64{
					podGroupKey: {
						"placement1": 1,
						"placement2": 2,
					},
				},
				// The pod can't be placed on the nominated node, so no new pod is scheduled on
				// placement2. PlacementFeasible still reports Success there (as it would when
				// minCount is already met by pods scheduled in previous cycles). Without the
				// anyScheduled guard the nominated placement would short-circuit and the gang
				// would land nowhere; the guard makes it fall back to placement1 (node1).
				filterStatus: map[string]*fwk.Status{
					nodes[1].Name: fwk.NewStatus(fwk.Unschedulable),
				},
			},
			placementFeasibleStatuses: [][]fwk.Code{{fwk.Success}, {fwk.Success}},
			nominatedNodeName:         nodes[1].Name,
			expectedHost:              nodes[0].Name,
		},
		"feasible nominated placement that schedules no new pod reports an actionable reason": {
			placementPlugin: fakePlacementPlugin{
				generatePlacementsResult: map[fwk.EntityKey]map[string][]string{
					podGroupKey: {
						"placement2": {nodes[1].Name},
					},
				},
				filterStatus: map[string]*fwk.Status{
					nodes[1].Name: fwk.NewStatus(fwk.Unschedulable),
				},
			},
			placementFeasibleStatuses: [][]fwk.Code{{fwk.Success}},
			nominatedNodeName:         nodes[1].Name,
			expectedStatus: fwk.NewStatus(
				fwk.Unschedulable,
				"0/1 placements are available, first placement status: nominated placement is feasible but no pending pod was scheduled",
			),
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
				tf.RegisterPlacementFeasiblePlugin(placementFeasiblePlugin.Name(), func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
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
			testPodGroup := &schedulingv1beta1.PodGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "pg", Namespace: "default"},
			}
			cache.AddGenericPodGroup(fwk.NewGenericPodGroup(testPodGroup))

			if err := cache.UpdateSnapshot(logger, snapshot); err != nil {
				t.Fatalf("Failed to update snapshot: %v", err)
			}

			pod := st.MakePod().Namespace("default").Name("foo").UID("foo").PodGroupName("pg").NominatedNodeName(tt.nominatedNodeName).Obj()
			queuedPodInfos := []*framework.QueuedPodInfo{{PodInfo: &framework.PodInfo{Pod: pod}}}
			childPodGroupInfo := &framework.PodGroupInfo{
				GenericPodGroup: fwk.NewGenericPodGroup(testPodGroup),
			}
			// The root is normally the pod group itself, but for the CPG case it is a separate
			// CompositePodGroup wrapping the child, which is what gates off the NNN fast path.
			rootPodGroupInfo := childPodGroupInfo
			if tt.rootIsCPG {
				rootPodGroupInfo = framework.NewPodGroupInfo(
					fwk.NewGenericCompositePodGroup(
						st.MakeCompositePodGroup().Name("pg").Namespace("default").Obj(),
					),
					childPodGroupInfo,
				)
			}
			pgInfo := newQueuedPodGroupInfo(rootPodGroupInfo, queuedPodInfos...)

			podGroupAlgorithm := testPodGroupAlgorithm(t, ctx, snapshot, cache, queue)
			result := podGroupAlgorithm.podGroupSchedulingPlacementAlgorithm(ctx, schedFwk, framework.NewCycleState(), childPodGroupInfo, pgInfo)

			if tt.expectError {
				if result.status == nil || !result.status.IsError() {
					t.Fatalf("expected an error status, got %v", result.status)
				}
				return
			}
			if tt.expectedStatus != nil {
				if diff := cmp.Diff(tt.expectedStatus, result.status, statusCmpOpt); diff != "" {
					t.Fatalf("unexpected status (-want,+got):\n%s", diff)
				}
				return
			}
			if result.status != nil && !result.status.IsSuccess() {
				t.Fatalf("expected the gang to be scheduled, got status %v", result.status)
			}
			if got := result.podResults[0].scheduleResult.SuggestedHost; got != tt.expectedHost {
				t.Fatalf("gang landed on %q, want %q", got, tt.expectedHost)
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
	podGroupPod := st.MakePod().Namespace("default").Name("foo").UID("foo").PodGroupName("pg").Obj()

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

				pgKeyVal := fwk.PodGroupKey("default", "pg")
				queuedPodInfos := []*framework.QueuedPodInfo{{PodInfo: &framework.PodInfo{Pod: podGroupPod}}}
				testPodGroup := st.MakePodGroup().Namespace("default").Name("pg").Obj()
				pgInfo := newQueuedPodGroupInfo(&framework.PodGroupInfo{GenericPodGroup: fwk.NewGenericPodGroup(testPodGroup)}, queuedPodInfos...)

				placementPlugin := fakePlacementPlugin{
					name: "FakeGeneratorPlugin",
					generatePlacementsResult: map[fwk.EntityKey]map[string][]string{
						pgKeyVal: placements,
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
						scorePlacementsResult: map[fwk.EntityKey]map[string]int64{
							pgKeyVal: placementScorePluginData.scorePlacementResult,
						},
						scorePlacementsStatus: map[fwk.EntityKey]map[string]*fwk.Status{
							pgKeyVal: placementScorePluginData.scorePlacementStatus,
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
				cache.AddGenericPodGroup(fwk.NewGenericPodGroup(testPodGroup))

				if err := cache.UpdateSnapshot(logger, snapshot); err != nil {
					t.Fatalf("Failed to update snapshot: %v", err)
				}

				podGroupAlgorithm := testPodGroupAlgorithm(t, ctx, snapshot, cache, queue)
				result := podGroupAlgorithm.podGroupSchedulingPlacementAlgorithm(ctx, schedFwk, framework.NewCycleState(), pgInfo.PodGroupInfo, pgInfo)

				expectedHost := placements[tt.expectedPlacement][0]
				actualHost := result.podResults[0].scheduleResult.SuggestedHost
				if expectedHost != actualHost {
					t.Fatalf("Unexpected algorithm result, expected placement %s with node %s, got node %s", tt.expectedPlacement, expectedHost, actualHost)
				}
			})
		}
	}
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
			podGroupPod := st.MakePod().Namespace("default").Name("foo").UID("foo").PodGroupName("pg").Obj()

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
			cache.AddGenericPodGroup(fwk.NewGenericPodGroup(testPodGroup))

			if err := cache.UpdateSnapshot(logger, snapshot); err != nil {
				t.Fatalf("Failed to update snapshot: %v", err)
			}

			queuedPodInfos := []*framework.QueuedPodInfo{{PodInfo: &framework.PodInfo{Pod: podGroupPod}}}
			pgInfo := newQueuedPodGroupInfo(&framework.PodGroupInfo{GenericPodGroup: fwk.NewGenericPodGroup(testPodGroup)}, queuedPodInfos...)
			podGroupAlgorithm := testPodGroupAlgorithm(t, ctx, snapshot, cache, queue)
			result := podGroupAlgorithm.podGroupSchedulingPlacementAlgorithm(ctx, schedFwk, framework.NewCycleState(), pgInfo.PodGroupInfo, pgInfo)
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

	leafPGInfo := &framework.PodGroupInfo{GenericPodGroup: fwk.NewGenericPodGroup(pg)}
	midPGInfo := framework.NewPodGroupInfo(fwk.NewGenericCompositePodGroup(midcpg), leafPGInfo)
	rootPGInfo := framework.NewPodGroupInfo(fwk.NewGenericCompositePodGroup(rootcpg), midPGInfo)

	logger, ctx := ktesting.NewTestContext(t)

	informerFactory := informers.NewSharedInformerFactory(clientsetfake.NewClientset(), 0)
	queue := internalqueue.NewSchedulingQueue(nil, informerFactory)

	tracker := &multiLevelPlacementStateTracker{
		generatePlacementsResult: map[fwk.EntityKey][]string{
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
		tf.RegisterPlacementFeasiblePlugin(tracker.Name(), func(_ context.Context, _ runtime.Object, h fwk.Handle) (fwk.Plugin, error) {
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
	cache.AddGenericPodGroup(fwk.NewGenericCompositePodGroup(rootcpg))
	cache.AddGenericPodGroup(fwk.NewGenericCompositePodGroup(midcpg))
	cache.AddGenericPodGroup(fwk.NewGenericPodGroup(pg))

	if err := cache.UpdateSnapshot(logger, snapshot); err != nil {
		t.Fatalf("Failed to update snapshot: %v", err)
	}

	cpgQueuedInfo := newQueuedPodGroupInfo(rootPGInfo, queuedPodInfo1)

	podGroupAlgorithm := testPodGroupAlgorithm(t, ctx, snapshot, cache, queue)
	results := podGroupAlgorithm.RunRootSchedulingAlgorithm(ctx, schedFwk, framework.NewCycleState(), cpgQueuedInfo)
	if result, ok := results[rootPGInfo.GetKey()]; !ok || !result.status.IsSuccess() {
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
	testRegistry := componentmetrics.NewKubeRegistry()
	testRegistry.MustRegister(metrics.GeneratedPlacementsTotal, metrics.PlacementEvaluations, metrics.PlacementEvaluationDuration)

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

	childPGInfo1 := &framework.PodGroupInfo{GenericPodGroup: fwk.NewGenericPodGroup(pg1)}
	childPGInfo2 := &framework.PodGroupInfo{GenericPodGroup: fwk.NewGenericPodGroup(pg2)}

	rootPGInfo := framework.NewPodGroupInfo(fwk.NewGenericCompositePodGroup(cpg), childPGInfo1, childPGInfo2)

	defaultPlacementResults := map[fwk.EntityKey]map[string][]string{
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
		placementPlugin               fakePlacementPlugin
		placementFeasibleStatuses     [][]fwk.Code
		expectedResults               map[fwk.EntityKey]PodGroupScheduledResult
		expectedGeneratedPlacements   map[string]int
		expectedFeasibleEvaluations   map[string]int
		expectedInfeasibleEvaluations map[string]int
	}{
		"respects higher score of parent placement": {
			placementPlugin: fakePlacementPlugin{
				generatePlacementsResult: defaultPlacementResults,
				scorePlacementsResult: map[fwk.EntityKey]map[string]int64{
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
			expectedResults: map[fwk.EntityKey]PodGroupScheduledResult{
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
			expectedGeneratedPlacements: map[string]int{
				metrics.CompositePodGroup: 2,
				metrics.PodGroup:          8,
			},
			expectedFeasibleEvaluations: map[string]int{
				metrics.CompositePodGroup: 2,
				metrics.PodGroup:          8,
			},
			expectedInfeasibleEvaluations: map[string]int{
				metrics.CompositePodGroup: 0,
				metrics.PodGroup:          0,
			},
		},
		"discards infeasible placements": {
			placementPlugin: fakePlacementPlugin{
				generatePlacementsResult: defaultPlacementResults,
				scorePlacementsResult: map[fwk.EntityKey]map[string]int64{
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
			expectedResults: map[fwk.EntityKey]PodGroupScheduledResult{
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
			expectedGeneratedPlacements: map[string]int{
				metrics.CompositePodGroup: 2,
				metrics.PodGroup:          4,
			},
			expectedFeasibleEvaluations: map[string]int{
				metrics.CompositePodGroup: 1,
				metrics.PodGroup:          4,
			},
			expectedInfeasibleEvaluations: map[string]int{
				metrics.CompositePodGroup: 1,
				metrics.PodGroup:          0,
			},
		},
		"returns unschedulable if no pods got scheduled": {
			placementPlugin: fakePlacementPlugin{
				generatePlacementsResult: defaultPlacementResults,
				scorePlacementsResult: map[fwk.EntityKey]map[string]int64{
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
			expectedResults: map[fwk.EntityKey]PodGroupScheduledResult{
				rootPGInfo.GetKey(): {
					status: fwk.NewStatus(fwk.Unschedulable, "no pods were schedulable"),
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
			expectedGeneratedPlacements: map[string]int{
				metrics.CompositePodGroup: 2,
				metrics.PodGroup:          8,
			},
			expectedFeasibleEvaluations: map[string]int{
				metrics.CompositePodGroup: 2,
				metrics.PodGroup:          8,
			},
			expectedInfeasibleEvaluations: map[string]int{
				metrics.CompositePodGroup: 0,
				metrics.PodGroup:          0,
			},
		},
		"returns unschedulable if no pods got scheduled and placement feasible rejected pods": {
			placementPlugin: fakePlacementPlugin{
				generatePlacementsResult: defaultPlacementResults,
				scorePlacementsResult: map[fwk.EntityKey]map[string]int64{
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
			placementFeasibleStatuses: [][]fwk.Code{
				// cpg, p1
				{fwk.Wait, fwk.Wait, fwk.Unschedulable},
				// pg1, p1
				{fwk.Wait, fwk.Unschedulable},
				// pg1, p2
				{fwk.Wait, fwk.Unschedulable},
				// pg2, p1
				{fwk.Wait, fwk.Unschedulable},
				// pg2, p2
				{fwk.Wait, fwk.Unschedulable},
				// cpg, p2
				{fwk.Wait, fwk.Wait, fwk.Unschedulable},
				// pg1, p3
				{fwk.Wait, fwk.Unschedulable},
				// pg1, p4
				{fwk.Wait, fwk.Unschedulable},
				// pg2, p3
				{fwk.Wait, fwk.Unschedulable},
				// pg2, p4
				{fwk.Wait, fwk.Unschedulable},
			},
			expectedResults: map[fwk.EntityKey]PodGroupScheduledResult{
				rootPGInfo.GetKey(): {
					status: fwk.NewStatus(fwk.Unschedulable, "0/2 placements are available, first placement status: injected placementFeasible status"),
				},
				childPGInfo1.GetKey(): {
					podResults: []algorithmResult{
						{
							podInfo: queuedPodInfo1,
							status:  fwk.NewStatus(fwk.Unschedulable, "0/1 nodes are available: 1 node1 rejected."),
						},
					},
					status: fwk.NewStatus(fwk.Unschedulable, "0/2 placements are available, first placement status: injected placementFeasible status"),
				},
				childPGInfo2.GetKey(): {
					podResults: []algorithmResult{
						{
							podInfo: queuedPodInfo2,
							status:  fwk.NewStatus(fwk.Unschedulable, "0/1 nodes are available: 1 node1 rejected."),
						},
					},
					status: fwk.NewStatus(fwk.Unschedulable, "0/2 placements are available, first placement status: injected placementFeasible status"),
				},
			},
			expectedGeneratedPlacements: map[string]int{
				metrics.CompositePodGroup: 2,
				metrics.PodGroup:          8,
			},
			expectedFeasibleEvaluations: map[string]int{
				metrics.CompositePodGroup: 0,
				metrics.PodGroup:          0,
			},
			expectedInfeasibleEvaluations: map[string]int{
				metrics.CompositePodGroup: 2,
				metrics.PodGroup:          8,
			},
		},
		"respects pods already scheduled in sibling pod groups": {
			placementPlugin: fakePlacementPlugin{
				generatePlacementsResult: defaultPlacementResults,
				podPerNode:               true,
				reservedNodes:            sets.New[string](),
				scorePlacementsResult: map[fwk.EntityKey]map[string]int64{
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
			expectedResults: map[fwk.EntityKey]PodGroupScheduledResult{
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
			expectedGeneratedPlacements: map[string]int{
				metrics.CompositePodGroup: 2,
				metrics.PodGroup:          8,
			},
			expectedFeasibleEvaluations: map[string]int{
				metrics.CompositePodGroup: 2,
				metrics.PodGroup:          8,
			},
			expectedInfeasibleEvaluations: map[string]int{
				metrics.CompositePodGroup: 0,
				metrics.PodGroup:          0,
			},
		},
		"when generate plugin fails at CPG, returns error": {
			placementPlugin: fakePlacementPlugin{
				generatePlacementsResult: defaultPlacementResults,
				generatePlacementsStatus: map[fwk.EntityKey]*fwk.Status{
					rootPGInfo.GetKey(): fwk.AsStatus(fmt.Errorf("injected error")),
				},
			},
			expectedResults: map[fwk.EntityKey]PodGroupScheduledResult{
				rootPGInfo.GetKey(): {
					status: fwk.NewStatus(fwk.Error, "injected error"),
				},
			},
			expectedGeneratedPlacements: map[string]int{
				metrics.CompositePodGroup: 0,
				metrics.PodGroup:          0,
			},
			expectedFeasibleEvaluations: map[string]int{
				metrics.CompositePodGroup: 0,
				metrics.PodGroup:          0,
			},
			expectedInfeasibleEvaluations: map[string]int{
				metrics.CompositePodGroup: 0,
				metrics.PodGroup:          0,
			},
		},
		"when generate plugin fails at PG, returns error": {
			placementPlugin: fakePlacementPlugin{
				generatePlacementsResult: defaultPlacementResults,
				generatePlacementsStatus: map[fwk.EntityKey]*fwk.Status{
					childPGInfo2.GetKey(): fwk.AsStatus(fmt.Errorf("injected error")),
				},
				scorePlacementsResult: map[fwk.EntityKey]map[string]int64{
					rootPGInfo.GetKey(): {
						"placement1": 1,
					},
					childPGInfo1.GetKey(): {
						"placement1": 2,
						"placement2": 1,
					},
					childPGInfo2.GetKey(): {
						"placement1": 2,
						"placement2": 1,
					},
				},
			},
			expectedResults: map[fwk.EntityKey]PodGroupScheduledResult{
				rootPGInfo.GetKey(): {
					status: fwk.NewStatus(fwk.Error, `composite pod group "cpg" evaluation failed due to child "pg2" error: injected error`),
				},
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
					status: fwk.AsStatus(fmt.Errorf("injected error")),
				},
			},
			expectedGeneratedPlacements: map[string]int{
				metrics.CompositePodGroup: 2,
				metrics.PodGroup:          2,
			},
			expectedFeasibleEvaluations: map[string]int{
				metrics.CompositePodGroup: 0,
				metrics.PodGroup:          2,
			},
			expectedInfeasibleEvaluations: map[string]int{
				metrics.CompositePodGroup: 0,
				metrics.PodGroup:          0,
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
				tf.RegisterPlacementFeasiblePlugin(placementFeasiblePlugin.Name(), func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
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
			cache.AddGenericPodGroup(fwk.NewGenericCompositePodGroup(cpg))
			cache.AddGenericPodGroup(fwk.NewGenericPodGroup(pg1))
			cache.AddGenericPodGroup(fwk.NewGenericPodGroup(pg2))
			cache.AddPodGroupMember(p1)
			cache.AddPodGroupMember(p2)

			if err := cache.UpdateSnapshot(logger, snapshot); err != nil {
				t.Fatalf("Failed to update snapshot: %v", err)
			}

			metrics.GeneratedPlacementsTotal.Reset()
			metrics.PlacementEvaluations.Reset()
			metrics.PlacementEvaluationDuration.Reset()

			rootPGInfo := framework.NewPodGroupInfo(fwk.NewGenericCompositePodGroup(cpg),
				&framework.PodGroupInfo{GenericPodGroup: fwk.NewGenericPodGroup(pg1)},
				&framework.PodGroupInfo{GenericPodGroup: fwk.NewGenericPodGroup(pg2)},
			)
			cpgInfo := newQueuedPodGroupInfo(rootPGInfo, queuedPodInfo1, queuedPodInfo2)

			podGroupAlgorithm := testPodGroupAlgorithm(t, ctx, snapshot, cache, queue)
			results := podGroupAlgorithm.RunRootSchedulingAlgorithm(ctx, schedFwk, framework.NewCycleState(), cpgInfo)
			gotResults := make(map[fwk.EntityKey]PodGroupScheduledResult, len(results))
			for k, v := range results {
				if v != nil {
					gotResults[k] = *v
				}
			}

			opts := cmp.Options{
				cmp.AllowUnexported(
					PodGroupScheduledResult{},
					algorithmResult{},
					ScheduleResult{},
					fwk.Status{},
					framework.PodInfo{}),
				cmpopts.IgnoreFields(PodGroupScheduledResult{}, "podGroupInfo", "placementCycleState", "anyScheduled"),
				cmpopts.IgnoreFields(algorithmResult{}, "podCtx", "schedulingDuration"),
				statusCmpOpt,
			}

			if diff := cmp.Diff(tt.expectedResults, gotResults, opts...); diff != "" {
				t.Fatalf("Unexpected algorithm results (-want,+got):\n%s", diff)
			}

			baseLabels := map[string]string{"profile": "test-scheduler"}
			for _, entityType := range []string{metrics.CompositePodGroup, metrics.PodGroup} {
				wantGen := tt.expectedGeneratedPlacements[entityType]
				wantFeas := tt.expectedFeasibleEvaluations[entityType]
				wantInfeas := tt.expectedInfeasibleEvaluations[entityType]

				fixedLabels := map[string]string{"profile": "test-scheduler", "entity": entityType}
				feasibleLabels := map[string]string{"profile": "test-scheduler", "result": metrics.FeasibleResult, "entity": entityType}
				infeasibleLabels := map[string]string{"profile": "test-scheduler", "result": metrics.InfeasibleResult, "entity": entityType}

				assertCounterValueFromGatherer(t, testRegistry, "scheduler_generated_placements_total", baseLabels, "entity", entityType, wantGen)
				assertCounterValueFromGatherer(t, testRegistry, "scheduler_placement_evaluations_total", fixedLabels, "result", metrics.FeasibleResult, wantFeas)
				assertCounterValueFromGatherer(t, testRegistry, "scheduler_placement_evaluations_total", fixedLabels, "result", metrics.InfeasibleResult, wantInfeas)

				assertHistogramSampleCountFromGatherer(t, testRegistry, "scheduler_placement_evaluation_duration_seconds", feasibleLabels, wantFeas)
				assertHistogramSampleCountFromGatherer(t, testRegistry, "scheduler_placement_evaluation_duration_seconds", infeasibleLabels, wantInfeas)
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

	childPGInfo1 := &framework.PodGroupInfo{GenericPodGroup: fwk.NewGenericPodGroup(pg1)}
	childPGInfo2 := &framework.PodGroupInfo{GenericPodGroup: fwk.NewGenericPodGroup(pg2)}

	rootPGInfo := framework.NewPodGroupInfo(fwk.NewGenericCompositePodGroup(cpg), childPGInfo1, childPGInfo2)

	placements := map[fwk.EntityKey]map[string][]string{
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
		scorePlacementsResult map[fwk.EntityKey]map[string]int64
		scorePlacementsStatus map[fwk.EntityKey]map[string]*fwk.Status
	}

	tests := map[string]struct {
		pluginData    []pluginData
		expectedHosts map[string]string
	}{
		"respects higher score of root placement1": {
			pluginData: []pluginData{
				{
					weight: 1,
					scorePlacementsResult: map[fwk.EntityKey]map[string]int64{
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
					scorePlacementsResult: map[fwk.EntityKey]map[string]int64{
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
					scorePlacementsResult: map[fwk.EntityKey]map[string]int64{
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
					scorePlacementsResult: map[fwk.EntityKey]map[string]int64{
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
				tf.RegisterPlacementFeasiblePlugin(gangscheduling.Name, gangPluginFactory),
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
			cache.AddGenericPodGroup(fwk.NewGenericCompositePodGroup(cpg))
			cache.AddGenericPodGroup(fwk.NewGenericPodGroup(pg1))
			cache.AddGenericPodGroup(fwk.NewGenericPodGroup(pg2))
			cache.AddPodGroupMember(p1)
			cache.AddPodGroupMember(p2)

			if err := cache.UpdateSnapshot(logger, snapshot); err != nil {
				t.Fatalf("Failed to update snapshot: %v", err)
			}

			rootPGInfo := framework.NewPodGroupInfo(fwk.NewGenericCompositePodGroup(cpg),
				&framework.PodGroupInfo{GenericPodGroup: fwk.NewGenericPodGroup(pg1)},
				&framework.PodGroupInfo{GenericPodGroup: fwk.NewGenericPodGroup(pg2)},
			)
			cpgInfo := newQueuedPodGroupInfo(rootPGInfo, queuedPodInfo1, queuedPodInfo2)

			podGroupAlgorithm := testPodGroupAlgorithm(t, ctx, snapshot, cache, queue)
			results := podGroupAlgorithm.RunRootSchedulingAlgorithm(ctx, schedFwk, framework.NewCycleState(), cpgInfo)
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
		results             map[string]map[string]*PodGroupScheduledResult
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
			results: map[string]map[string]*PodGroupScheduledResult{
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
			results: map[string]map[string]*PodGroupScheduledResult{
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
			results: map[string]map[string]*PodGroupScheduledResult{
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
			results: map[string]map[string]*PodGroupScheduledResult{
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
				var gpg *fwk.GenericPodGroup
				if len(children) > 0 {
					cpg := st.MakeCompositePodGroup().Name(node.name).Namespace("default").Obj()
					gpg = fwk.NewGenericCompositePodGroup(cpg)
				} else {
					pg := st.MakePodGroup().Name(node.name).Namespace("default").Obj()
					gpg = fwk.NewGenericPodGroup(pg)
				}
				pgi := framework.NewPodGroupInfo(gpg, children...)
				nameToKey[node.name] = pgi.GetKey()
				return pgi
			}

			root := buildTree(tc.tree)

			successfulResults := make(map[*fwk.Placement]map[fwk.EntityKey]*PodGroupScheduledResult)
			for placementName, resMap := range tc.results {
				placement := &fwk.Placement{Name: placementName}
				entityMap := make(map[fwk.EntityKey]*PodGroupScheduledResult, len(resMap))
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

			podGroupAlgorithm := testPodGroupAlgorithm(t, ctx, snapshot, nil, queue)
			_, status := podGroupAlgorithm.findBestCompositePodGroupPlacement(ctx, schedFwk, framework.NewCycleState(), root, successfulResults)
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

func TestBuildPodGroupSchedulingFunc(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.TopologyAwareWorkloadScheduling: true,
		features.GenericWorkload:                 true,
		features.CompositePodGroup:               true,
	})

	// Tree structure:
	//                      cpg-root (Gang, MinGroupCount: 2)
	//                  /                  |                   \
	//            midcpg1               midcpg2               midcpg3
	//      (Gang, MinGroup: 1)   (Gang, MinGroup: 2)   (Gang, MinGroup: 1)
	//              |                   /     \                   |
	//             pg1                pg2     pg3                pg4
	//           (Min: 1)          (Min: 1) (Min: 1)           (Min: 1)
	//             p1                 p2      p3                 p4
	tests := []struct {
		name             string
		filterStatus     map[string]*fwk.Status
		wantStatusCode   fwk.Code
		wantAssignedPods []string
	}{
		{
			name: "all subtrees succeed and preserve tree-walk order",
			filterStatus: map[string]*fwk.Status{
				"p1": nil,
				"p2": nil,
				"p3": nil,
				"p4": nil,
			},
			wantStatusCode:   fwk.Success,
			wantAssignedPods: []string{"p1", "p2", "p3", "p4"},
		},
		{
			name: "excludes pods from failed intermediate CompositePodGroup and preserves tree-walk order",
			filterStatus: map[string]*fwk.Status{
				"p1": nil,
				"p2": nil,
				"p3": fwk.NewStatus(fwk.Unschedulable, "insufficient cpu"),
				"p4": nil,
			},
			wantStatusCode:   fwk.Success,
			wantAssignedPods: []string{"p1", "p4"},
		},
		{
			name: "returns nil assignments when root CompositePodGroup fails",
			filterStatus: map[string]*fwk.Status{
				"p1": nil,
				"p2": nil,
				"p3": fwk.NewStatus(fwk.Unschedulable, "insufficient cpu"),
				"p4": fwk.NewStatus(fwk.Unschedulable, "insufficient cpu"),
			},
			wantStatusCode:   fwk.Unschedulable,
			wantAssignedPods: nil,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			const ns = "default"
			testNode := st.MakeNode().Name("node1").UID("node1").Capacity(map[v1.ResourceName]string{
				v1.ResourceCPU:    "100",
				v1.ResourceMemory: "100Gi",
				v1.ResourcePods:   "100",
			}).Obj()
			testNode.Status.Allocatable = testNode.Status.Capacity

			cache := internalcache.New(ctx, nil, true, true /* CompositePodGroup */)
			cache.AddNode(logger, testNode)

			rootCPG := st.MakeCompositePodGroup().Name("cpg-root").Namespace(ns).MinGroupCount(2).Obj()
			midCPG1 := st.MakeCompositePodGroup().Name("midcpg1").Namespace(ns).MinGroupCount(1).ParentCompositePodGroup("cpg-root").Obj()
			midCPG2 := st.MakeCompositePodGroup().Name("midcpg2").Namespace(ns).MinGroupCount(2).ParentCompositePodGroup("cpg-root").Obj()
			midCPG3 := st.MakeCompositePodGroup().Name("midcpg3").Namespace(ns).MinGroupCount(1).ParentCompositePodGroup("cpg-root").Obj()
			allCPGs := []*schedulingv1beta1.CompositePodGroup{rootCPG, midCPG1, midCPG2, midCPG3}
			for _, cpg := range allCPGs {
				cache.AddGenericPodGroup(fwk.NewGenericCompositePodGroup(cpg))
			}

			pg1 := st.MakePodGroup().Name("pg1").Namespace(ns).MinCount(1).ParentCompositePodGroup("midcpg1").Obj()
			pg2 := st.MakePodGroup().Name("pg2").Namespace(ns).MinCount(1).ParentCompositePodGroup("midcpg2").Obj()
			pg3 := st.MakePodGroup().Name("pg3").Namespace(ns).MinCount(1).ParentCompositePodGroup("midcpg2").Obj()
			pg4 := st.MakePodGroup().Name("pg4").Namespace(ns).MinCount(1).ParentCompositePodGroup("midcpg3").Obj()
			allPGs := []*schedulingv1beta1.PodGroup{pg1, pg2, pg3, pg4}
			for _, pg := range allPGs {
				cache.AddGenericPodGroup(fwk.NewGenericPodGroup(pg))
			}

			p1 := st.MakePod().Namespace(ns).Name("p1").UID("p1").PodGroupName("pg1").Priority(100).SchedulerName("test-scheduler").Obj()
			p2 := st.MakePod().Namespace(ns).Name("p2").UID("p2").PodGroupName("pg2").Priority(100).SchedulerName("test-scheduler").Obj()
			p3 := st.MakePod().Namespace(ns).Name("p3").UID("p3").PodGroupName("pg3").Priority(100).SchedulerName("test-scheduler").Obj()
			p4 := st.MakePod().Namespace(ns).Name("p4").UID("p4").PodGroupName("pg4").Priority(100).SchedulerName("test-scheduler").Obj()
			allPods := []*v1.Pod{p1, p2, p3, p4}
			for _, pod := range allPods {
				cache.AddPodGroupMember(pod)
			}

			podGroupInfo := buildHierarchicalQueuedPodGroupInfo(rootCPG, allCPGs, allPGs, allPods)

			fakePlugin := &fakePodGroupPlugin{
				filterStatus: tc.filterStatus,
			}
			gangPluginFactory := func(ctx context.Context, obj runtime.Object, handle fwk.Handle) (fwk.Plugin, error) {
				return gangscheduling.New(ctx, obj, handle, feature.Features{
					EnableTopologyAwareWorkloadScheduling: true,
					EnableCompositePodGroup:               true,
				})
			}

			registry := []tf.RegisterPluginFunc{
				tf.RegisterFilterPlugin(fakePlugin.Name(), func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
					return fakePlugin, nil
				}),
				tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
				tf.RegisterPlacementFeasiblePlugin(gangscheduling.Name, gangPluginFactory),
			}

			clientObjs := []runtime.Object{testNode, rootCPG, midCPG1, midCPG2, midCPG3, pg1, pg2, pg3, pg4, p1, p2, p3, p4}
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
			if err != nil {
				t.Fatalf("Failed to create framework: %v", err)
			}

			if err := cache.UpdateSnapshot(logger, snapshot); err != nil {
				t.Fatalf("Failed to update snapshot: %v", err)
			}

			podGroupAlgorithm := testPodGroupAlgorithm(t, ctx, snapshot, cache, queue)
			pgSchedulingFunc := podGroupAlgorithm.BuildPodGroupSchedulingFunc(schedFwk, podGroupInfo)
			// Run pgSchedulingFunc multiple times to verify that ProposedAssignments
			// returns pods in deterministic tree-walk order instead of random map order.
			for range 20 {
				assignments, status := pgSchedulingFunc(ctx)
				if status.Code() != tc.wantStatusCode {
					t.Fatalf("pgSchedulingFunc status code = %v, want %v (status: %v)", status.Code(), tc.wantStatusCode, status)
				}
				if tc.wantAssignedPods == nil {
					if assignments != nil {
						t.Fatalf("pgSchedulingFunc assignments = %v, want nil", assignments)
					}
					continue
				}
				if assignments == nil {
					t.Fatalf("pgSchedulingFunc returned nil assignments, want %v", tc.wantAssignedPods)
				}
				gotPods := make([]string, 0, len(assignments.ProposedAssignments))
				for _, pa := range assignments.ProposedAssignments {
					gotPods = append(gotPods, pa.GetPod().Name)
				}
				if diff := cmp.Diff(tc.wantAssignedPods, gotPods); diff != "" {
					t.Fatalf("Unexpected proposed assignments (-want,+got):\n%s", diff)
				}
			}
		})
	}
}

func TestNumFeasiblePlacementsToFind(t *testing.T) {
	makePlacements := func(placementsCount, nodesCount int) []*fwk.Placement {
		res := make([]*fwk.Placement, placementsCount)
		j := 0
		for i := range placementsCount {
			nodes := make([]fwk.NodeInfo, 0)
			for ; j < nodesCount*(i+1)/placementsCount; j++ {
				nodes = append(nodes, framework.NewNodeInfo())
			}
			res[i] = &fwk.Placement{
				Nodes: nodes,
			}
		}
		return res
	}

	tests := []struct {
		name              string
		globalPercentage  int32
		profilePercentage *int32
		placementsCount   int
		nodesCount        int
		wantNumPlacements int
	}{
		{
			name:              "default limit for small cluster and small number of placements",
			placementsCount:   10,
			nodesCount:        100,
			wantNumPlacements: 9,
		},
		{
			name:              "default limit for large cluster and small number of placements",
			placementsCount:   20,
			nodesCount:        5000,
			wantNumPlacements: 2,
		},
		{
			name:              "default limit for small cluster and large number of placements",
			placementsCount:   100,
			nodesCount:        100,
			wantNumPlacements: 99,
		},
		{
			name:              "default limit for large cluster and large number of placements",
			placementsCount:   1000,
			nodesCount:        5000,
			wantNumPlacements: 100,
		},
		{
			name:              "default limit does not drop below 5%",
			placementsCount:   10000,
			nodesCount:        50000,
			wantNumPlacements: 500,
		},
		{
			name:              "non-default limit can drop below 5%",
			globalPercentage:  1,
			placementsCount:   10000,
			nodesCount:        50000,
			wantNumPlacements: 100,
		},
		{
			name:              "limit cannot drop below 1 placement",
			globalPercentage:  1,
			placementsCount:   10,
			nodesCount:        50000,
			wantNumPlacements: 1,
		},
		{
			name:              "both global and profile limit set, profile limit lower than global",
			globalPercentage:  50,
			profilePercentage: ptr.To[int32](5),
			placementsCount:   1000,
			nodesCount:        5000,
			wantNumPlacements: 50,
		},
		{
			name:              "both global and profile limit set, profile limit higher than global",
			globalPercentage:  50,
			profilePercentage: ptr.To[int32](55),
			placementsCount:   1000,
			nodesCount:        5000,
			wantNumPlacements: 550,
		},
		{
			name:              "both global and profile limit set, profile limit using default",
			globalPercentage:  50,
			profilePercentage: ptr.To[int32](0), // use default limit
			placementsCount:   1000,
			nodesCount:        5000,
			wantNumPlacements: 100,
		},
		{
			name:              "only profile limit set",
			profilePercentage: ptr.To[int32](5),
			placementsCount:   1000,
			nodesCount:        5000,
			wantNumPlacements: 50,
		},
		{
			name:              "only global limit set",
			globalPercentage:  50,
			placementsCount:   1000,
			nodesCount:        5000,
			wantNumPlacements: 500,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			podGroupAlgorithm := &PodGroupSchedulingAlgorithm{
				percentageOfPlacementsToScore: tt.globalPercentage,
			}
			placements := makePlacements(tt.placementsCount, tt.nodesCount)
			if got := podGroupAlgorithm.numFeasiblePlacementsToFind(tt.profilePercentage, placements); got != tt.wantNumPlacements {
				t.Errorf("PodGroupSchedulingAlgorithm.numFeasiblePlacementsToFind() = %v, want %v", got, tt.wantNumPlacements)
			}
		})
	}
}

type trackingPlacementScorePlugin struct {
	lock             sync.Mutex
	scoredPlacements sets.Set[string]
}

var _ fwk.PlacementScorePlugin = &trackingPlacementScorePlugin{}

func (t *trackingPlacementScorePlugin) Name() string {
	return "trackingPlacementScorePlugin"
}

func (t *trackingPlacementScorePlugin) PlacementScoreExtensions() fwk.PlacementScoreExtensions {
	return nil
}

func (t *trackingPlacementScorePlugin) ScorePlacement(ctx context.Context, state fwk.PlacementCycleState, podGroup fwk.PodGroupInfo, placement *fwk.PodGroupAssignments) (int64, *fwk.Status) {
	t.lock.Lock()
	defer t.lock.Unlock()
	t.scoredPlacements.Insert(placement.Name)
	return 1, nil
}

type trackingFilterPlugin struct {
	lock        sync.Mutex
	scoredNodes sets.Set[string]
}

var _ fwk.FilterPlugin = &trackingFilterPlugin{}

func (t *trackingFilterPlugin) Name() string {
	return "trackingFilterPlugin"
}

func (t *trackingFilterPlugin) Filter(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeInfo fwk.NodeInfo) *fwk.Status {
	t.lock.Lock()
	defer t.lock.Unlock()
	t.scoredNodes.Insert(nodeInfo.Node().Name)
	return nil
}

type scheduledPodPlacementFeasiblePlugin struct{}

var _ fwk.PlacementFeasiblePlugin = &scheduledPodPlacementFeasiblePlugin{}

func (p *scheduledPodPlacementFeasiblePlugin) Name() string {
	return names.GangScheduling
}

func (p *scheduledPodPlacementFeasiblePlugin) PlacementFeasible(_ context.Context, _ fwk.PlacementCycleState, _ fwk.PodGroupInfo, progress fwk.PlacementProgress) *fwk.Status {
	if progress.Remaining > 0 {
		return fwk.NewStatus(fwk.Wait)
	}
	if progress.Scheduled == 0 {
		return fwk.NewStatus(fwk.Unschedulable, "no pods scheduled")
	}
	return nil
}

func TestPodGroupSchedulingPlacementAlgorithm_PlacementLimit(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.TopologyAwareWorkloadScheduling: true,
		features.GenericWorkload:                 true,
	})

	tests := []struct {
		name                         string
		numFeasiblePlacements        int
		numInfeasiblePlacements      int
		percentageLimit              int32
		skipScore                    bool
		expectedNumScoredPlacements  int
		expectedNumCheckedPlacements *int // nil if cannot be determined due to random placement order
	}{
		{
			name:                        "Only limits the feasible placements",
			numFeasiblePlacements:       3,
			numInfeasiblePlacements:     97,
			percentageLimit:             2,
			expectedNumScoredPlacements: 2,
		},
		{
			name:                         "Checks all placements in case there is no feasible placement",
			numFeasiblePlacements:        0,
			numInfeasiblePlacements:      100,
			percentageLimit:              40,
			expectedNumScoredPlacements:  0,
			expectedNumCheckedPlacements: new(100),
		},
		{
			name:                         "Does not score placements if limit is 1",
			numFeasiblePlacements:        100,
			numInfeasiblePlacements:      0,
			percentageLimit:              1,
			expectedNumScoredPlacements:  0,
			expectedNumCheckedPlacements: new(1),
		},
		{
			name:                         "Stops after a single placement if there are no score plugins",
			numFeasiblePlacements:        100,
			numInfeasiblePlacements:      0,
			percentageLimit:              40,
			skipScore:                    true,
			expectedNumCheckedPlacements: new(1),
		},
	}

	for _, test := range tests {
		for _, compositePodGroup := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s (CPG: %v)", test.name, compositePodGroup), func(t *testing.T) {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CompositePodGroup, compositePodGroup)
				logger, ctx := ktesting.NewTestContext(t)

				informerFactory := informers.NewSharedInformerFactory(clientsetfake.NewClientset(), 0)
				queue := internalqueue.NewSchedulingQueue(nil, informerFactory)

				numNodes := test.numFeasiblePlacements + test.numInfeasiblePlacements
				nodes := make([]*v1.Node, numNodes)
				nodeNames := make([]string, numNodes)

				podGroupKey := fwk.PodGroupKey("default", "pg")
				// The limit applies to the placements of the root group. A PodGroup under a
				// CompositePodGroup gets a single placement within each root placement.
				rootKey := podGroupKey
				if compositePodGroup {
					rootKey = fwk.CompositePodGroupKey("default", "cpg")
				}
				placementPlugin := &fakePlacementPlugin{
					name: "fakePlacementPlugin",
					generatePlacementsResult: map[fwk.EntityKey]map[string][]string{
						rootKey: {},
					},
					filterStatus: make(map[string]*fwk.Status),
				}

				for i := range numNodes {
					nodeName := fmt.Sprintf("n%v", i)
					nodes[i] = st.MakeNode().Name(nodeName).UID(nodeName).Obj()
					nodeNames[i] = nodeName
					placementName := fmt.Sprintf("p%v", i)
					placementPlugin.generatePlacementsResult[rootKey][placementName] = []string{nodeName}
					placementPlugin.filterStatus[nodeName] = fwk.NewStatus(fwk.Unschedulable)
				}
				if compositePodGroup {
					placementPlugin.generatePlacementsResult[podGroupKey] = map[string][]string{"pg": nodeNames}
				}

				feasiblePlacements := sets.New[string]()
				for placement, nodeNames := range placementPlugin.generatePlacementsResult[rootKey] {
					if len(feasiblePlacements) == test.numFeasiblePlacements {
						break
					}
					placementPlugin.filterStatus[nodeNames[0]] = fwk.NewStatus(fwk.Success)
					feasiblePlacements.Insert(placement)
				}

				trackingFilter := &trackingFilterPlugin{scoredNodes: sets.New[string]()}
				trackingScore := &trackingPlacementScorePlugin{scoredPlacements: sets.New[string]()}
				placementFeasible := &scheduledPodPlacementFeasiblePlugin{}

				registry := []tf.RegisterPluginFunc{
					tf.RegisterPlacementGeneratePlugin(placementPlugin.Name(), func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
						return placementPlugin, nil
					}),
					tf.RegisterFilterPlugin(trackingFilter.Name(), func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
						return trackingFilter, nil
					}),
					tf.RegisterFilterPlugin(placementPlugin.Name(), func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
						return placementPlugin, nil
					}),
					tf.RegisterPlacementFeasiblePlugin(placementFeasible.Name(), func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
						return placementFeasible, nil
					}),
				}

				if !test.skipScore {
					registry = append(registry,
						tf.RegisterPlacementScorePlugin(trackingScore.Name(), func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
							return trackingScore, nil
						}, 1),
					)
				}

				cache := internalcache.New(ctx, nil, true, compositePodGroup)
				for _, node := range nodes {
					cache.AddNode(logger, node)
				}
				testPodGroup := st.MakePodGroup().Name("pg").Namespace("default").Obj()
				testCompositePodGroup := st.MakeCompositePodGroup().Name("cpg").Namespace("default").Obj()
				cache.AddGenericPodGroup(fwk.NewGenericPodGroup(testPodGroup))
				if compositePodGroup {
					cache.AddGenericPodGroup(fwk.NewGenericCompositePodGroup(testCompositePodGroup))
				}
				snapshot := internalcache.NewEmptySnapshot()
				if err := cache.UpdateSnapshot(logger, snapshot); err != nil {
					t.Fatalf("Failed to update snapshot: %v", err)
				}

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

				podGroupAlgorithm := testPodGroupAlgorithm(t, ctx, snapshot, cache, queue)
				podGroupAlgorithm.percentageOfPlacementsToScore = test.percentageLimit

				podGroupPod := st.MakePod().Name("foo").Namespace("default").UID("foo").PodGroupName("pg").Obj()
				runPlacementAlgorithm(ctx, t, podGroupAlgorithm, schedFwk, testPodGroup, testCompositePodGroup, compositePodGroup, podGroupPod)

				if got := len(trackingScore.scoredPlacements); got != test.expectedNumScoredPlacements {
					t.Errorf("Unexpected number of scored placements, want %d, got %d", test.expectedNumScoredPlacements, got)
				}

				if len(trackingScore.scoredPlacements) > 0 {
					for skippedPlacement := range feasiblePlacements.Difference(trackingScore.scoredPlacements) {
						placementNode := placementPlugin.generatePlacementsResult[rootKey][skippedPlacement][0]
						if trackingFilter.scoredNodes.Has(placementNode) {
							t.Errorf("Unexpected checked node for skipped placement %s, want none, got %v", skippedPlacement, placementNode)
						}
					}
				}

				if test.expectedNumCheckedPlacements != nil {
					// 1 unique node per placement
					if got := len(trackingFilter.scoredNodes); got != *test.expectedNumCheckedPlacements {
						t.Errorf("Unexpected number of checked placements, want %d, got %d", *test.expectedNumCheckedPlacements, got)
					}
				}
			})
		}
	}
}

// runPlacementAlgorithm runs the placement algorithm for podGroup, or for a CompositePodGroup with
// podGroup as its only child when compositePodGroup is true.
func runPlacementAlgorithm(ctx context.Context, t *testing.T, podGroupAlgorithm *PodGroupSchedulingAlgorithm, schedFwk framework.Framework, podGroup *schedulingv1beta1.PodGroup, cpg *schedulingv1beta1.CompositePodGroup, compositePodGroup bool, pod *v1.Pod) *PodGroupScheduledResult {
	t.Helper()
	podGroupInfo := &framework.PodGroupInfo{GenericPodGroup: fwk.NewGenericPodGroup(podGroup)}
	podInfo := &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: pod}}
	if compositePodGroup {
		cpgInfo := newQueuedPodGroupInfo(
			framework.NewPodGroupInfo(fwk.NewGenericCompositePodGroup(cpg), podGroupInfo),
			podInfo,
		)
		return podGroupAlgorithm.compositePodGroupSchedulingPlacementAlgorithm(ctx, schedFwk, framework.NewCycleState(), cpgInfo, cpgInfo.PodGroupInfo, map[fwk.EntityKey]*PodGroupScheduledResult{})
	}
	pgInfo := newQueuedPodGroupInfo(podGroupInfo, podInfo)
	return podGroupAlgorithm.podGroupSchedulingPlacementAlgorithm(ctx, schedFwk, framework.NewCycleState(), pgInfo.PodGroupInfo, pgInfo)
}

// fixedOrderPlacementPlugin returns one placement per node of the parent placement in a
// deterministic order, so a test can verify that the scheduler itself randomizes the evaluation
// order. It records the first slice it returns to verify that the scheduler doesn't mutate it.
type fixedOrderPlacementPlugin struct {
	placementNames   []string
	nodePerPlacement map[string]string
	firstPlacements  []*fwk.Placement
}

func (p *fixedOrderPlacementPlugin) Name() string { return "fixedOrderPlacementPlugin" }

func (p *fixedOrderPlacementPlugin) GeneratePlacements(ctx context.Context, state fwk.PodGroupCycleState, podGroup fwk.PodGroupInfo, parentPlacement *fwk.Placement) (*fwk.GeneratePlacementsResult, *fwk.Status) {
	parentNodes := map[string]fwk.NodeInfo{}
	for _, node := range parentPlacement.Nodes {
		parentNodes[node.Node().Name] = node
	}
	placements := make([]*fwk.Placement, 0, len(p.placementNames))
	for _, name := range p.placementNames {
		if node, ok := parentNodes[p.nodePerPlacement[name]]; ok {
			placements = append(placements, &fwk.Placement{
				Name:  name,
				Nodes: []fwk.NodeInfo{node},
			})
		}
	}
	if p.firstPlacements == nil {
		p.firstPlacements = placements
	}
	return &fwk.GeneratePlacementsResult{Placements: placements}, nil
}

func TestPodGroupSchedulingPlacementAlgorithm_UsesShufflePlacements(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.TopologyAwareWorkloadScheduling: true,
		features.GenericWorkload:                 true,
	})

	const numPlacements = 10
	// The test shuffler moves this placement to the front.
	const shuffledFirstPlacement = "p7"
	allNodes := sets.New[string]()
	for i := range numPlacements {
		allNodes.Insert(fmt.Sprintf("n%v", i))
	}

	tests := []struct {
		name              string
		compositePodGroup bool
		percentageLimit   int32
		nominatedNodeName string
		wantCheckedNodes  sets.Set[string]
		wantShuffled      bool
	}{
		{
			name:             "first shuffled placement",
			percentageLimit:  10,
			wantCheckedNodes: sets.New("n7"),
			wantShuffled:     true,
		},
		{
			name:              "nominated placement is used without shuffling",
			percentageLimit:   10,
			nominatedNodeName: "n0",
			wantCheckedNodes:  sets.New("n0"),
		},
		{
			name:             "no shuffling when every placement may be scored",
			percentageLimit:  100,
			wantCheckedNodes: allNodes,
		},
		{
			name:              "first shuffled placement of a CompositePodGroup",
			compositePodGroup: true,
			percentageLimit:   10,
			wantCheckedNodes:  sets.New("n7"),
			wantShuffled:      true,
		},
		{
			name:              "no shuffling of CompositePodGroup placements when every placement may be scored",
			compositePodGroup: true,
			percentageLimit:   100,
			wantCheckedNodes:  allNodes,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CompositePodGroup, test.compositePodGroup)
			logger, ctx := ktesting.NewTestContext(t)

			placementPlugin := &fixedOrderPlacementPlugin{nodePerPlacement: map[string]string{}}
			nodes := make([]*v1.Node, numPlacements)
			for i := range numPlacements {
				nodeName := fmt.Sprintf("n%v", i)
				placementName := fmt.Sprintf("p%v", i)
				nodes[i] = st.MakeNode().Name(nodeName).UID(nodeName).Obj()
				placementPlugin.placementNames = append(placementPlugin.placementNames, placementName)
				placementPlugin.nodePerPlacement[placementName] = nodeName
			}

			informerFactory := informers.NewSharedInformerFactory(clientsetfake.NewClientset(), 0)
			queue := internalqueue.NewSchedulingQueue(nil, informerFactory)

			cache := internalcache.New(ctx, nil, true, test.compositePodGroup)
			for _, node := range nodes {
				cache.AddNode(logger, node)
			}
			testPodGroup := st.MakePodGroup().Name("pg").Namespace("default").Obj()
			testCompositePodGroup := st.MakeCompositePodGroup().Name("cpg").Namespace("default").Obj()
			cache.AddGenericPodGroup(fwk.NewGenericPodGroup(testPodGroup))
			if test.compositePodGroup {
				cache.AddGenericPodGroup(fwk.NewGenericCompositePodGroup(testCompositePodGroup))
			}
			snapshot := internalcache.NewEmptySnapshot()
			if err := cache.UpdateSnapshot(logger, snapshot); err != nil {
				t.Fatalf("Failed to update snapshot: %v", err)
			}

			trackingFilter := &trackingFilterPlugin{scoredNodes: sets.New[string]()}
			trackingScore := &trackingPlacementScorePlugin{scoredPlacements: sets.New[string]()}
			registry := []tf.RegisterPluginFunc{
				tf.RegisterPlacementGeneratePlugin(placementPlugin.Name(), func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
					return placementPlugin, nil
				}),
				tf.RegisterFilterPlugin(trackingFilter.Name(), func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
					return trackingFilter, nil
				}),
				tf.RegisterPlacementScorePlugin(trackingScore.Name(), func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
					return trackingScore, nil
				}, 1),
				tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			}

			schedFwk, err := tf.NewFramework(ctx, registry, "test-scheduler",
				frameworkruntime.WithInformerFactory(informerFactory),
				frameworkruntime.WithSnapshotSharedLister(snapshot),
				frameworkruntime.WithPodNominator(queue),
			)
			if err != nil {
				t.Fatalf("Failed to create new framework: %v", err)
			}

			shuffled := false
			podGroupAlgorithm := testPodGroupAlgorithm(t, ctx, snapshot, cache, queue)
			podGroupAlgorithm.percentageOfPlacementsToScore = test.percentageLimit
			podGroupAlgorithm.shufflePlacements = func(placements []*fwk.Placement) {
				shuffled = true
				for i, p := range placements {
					if p.Name == shuffledFirstPlacement {
						placements[0], placements[i] = placements[i], placements[0]
						return
					}
				}
			}

			podGroupPod := st.MakePod().Name("foo").Namespace("default").UID("foo").PodGroupName("pg").NominatedNodeName(test.nominatedNodeName).Obj()
			result := runPlacementAlgorithm(ctx, t, podGroupAlgorithm, schedFwk, testPodGroup, testCompositePodGroup, test.compositePodGroup, podGroupPod)
			if !result.status.IsSuccess() {
				t.Fatalf("Expected successful placement, got %v", result.status)
			}

			if diff := cmp.Diff(sets.List(test.wantCheckedNodes), sets.List(trackingFilter.scoredNodes)); diff != "" {
				t.Errorf("Unexpected checked nodes (-want,+got):\n%s", diff)
			}
			if shuffled != test.wantShuffled {
				t.Errorf("Unexpected shuffle call, want %v, got %v", test.wantShuffled, shuffled)
			}
			var gotPluginOrder []string
			for _, p := range placementPlugin.firstPlacements {
				gotPluginOrder = append(gotPluginOrder, p.Name)
			}
			if diff := cmp.Diff(placementPlugin.placementNames, gotPluginOrder); diff != "" {
				t.Errorf("Placements returned by the plugin were mutated (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestNewShufflePlacementsRandomizesOrder(t *testing.T) {
	client := clientsetfake.NewClientset()
	informerFactory := informers.NewSharedInformerFactory(client, 0)
	eventBroadcaster := events.NewBroadcaster(&events.EventSinkImpl{Interface: client.EventsV1()})
	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	sched, err := New(ctx, client, informerFactory, nil, profile.NewRecorderFactory(eventBroadcaster))
	if err != nil {
		t.Fatalf("Failed to create scheduler: %v", err)
	}
	if sched.podGroupAlgorithm.shufflePlacements == nil {
		t.Fatal("Expected New to set a non-nil shufflePlacements function")
	}

	const numPlacements = 10
	const runs = 50
	firstNames := sets.New[string]()
	for range runs {
		placements := make([]*fwk.Placement, numPlacements)
		for i := range placements {
			placements[i] = &fwk.Placement{Name: fmt.Sprintf("p%d", i)}
		}
		sched.podGroupAlgorithm.shufflePlacements(placements)
		firstNames.Insert(placements[0].Name)
	}

	if len(firstNames) <= 1 {
		t.Errorf("Expected the default shuffler to randomize placement order, but saw only %d distinct first placement(s) over %d runs", len(firstNames), runs)
	}
}

// fakeAssumeAndReserveRecorderPlugin records every assume and revert in order.
type fakeAssumeAndReserveRecorderPlugin struct {
	mu     sync.Mutex
	events []string
	// assumeErrorAt is a "pod@node" whose assume returns an error, but only after it has been previously reverted,
	// because the first assume of a pod on a node is done during placement evaluation, which is always reverted.
	// Skipping it makes the error hit the assume of the chosen placement for subsequent siblings.
	assumeErrorAt string
}

var _ fwk.ReservePlugin = &fakeAssumeAndReserveRecorderPlugin{}

func (p *fakeAssumeAndReserveRecorderPlugin) Name() string { return "FakeAssumeAndReserveRecorder" }

func (p *fakeAssumeAndReserveRecorderPlugin) Reserve(_ context.Context, _ fwk.CycleState, pod *v1.Pod, nodeName string) *fwk.Status {
	p.mu.Lock()
	defer p.mu.Unlock()
	key := pod.Name + "@" + nodeName
	revertedBefore := slices.Contains(p.events, "revert "+key)
	p.events = append(p.events, "assume "+key)
	if revertedBefore && key == p.assumeErrorAt {
		return fwk.NewStatus(fwk.Error, "injected reserve failure")
	}
	return nil
}

func (p *fakeAssumeAndReserveRecorderPlugin) Unreserve(_ context.Context, _ fwk.CycleState, pod *v1.Pod, nodeName string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.events = append(p.events, "revert "+pod.Name+"@"+nodeName)
}

func TestRunRootSchedulingAlgorithm_AssumeSubtrees(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.GenericWorkload:                 true,
		features.TopologyAwareWorkloadScheduling: true,
		features.CompositePodGroup:               true,
	})

	rootPodGroup := st.MakePodGroup().Name("pg").Namespace("default").MinCount(1).Obj()
	podGroupPod := st.MakePod().Name("pod").Namespace("default").UID("pod").PodGroupName("pg").Obj()

	rootCompositePodGroup := st.MakeCompositePodGroup().Name("cpg").Namespace("default").Obj()
	childBasicPodGroup1 := st.MakePodGroup().Name("pg1").Namespace("default").ParentCompositePodGroup("cpg").BasicPolicy().Obj()
	childPodGroup1 := st.MakePodGroup().Name("pg1").Namespace("default").ParentCompositePodGroup("cpg").MinCount(1).Obj()
	childPodGroup2 := st.MakePodGroup().Name("pg2").Namespace("default").ParentCompositePodGroup("cpg").MinCount(1).Obj()
	pg1Pod1 := st.MakePod().Name("pg1-pod1").Namespace("default").UID("pg1-pod1").PodGroupName("pg1").Obj()
	pg1Pod2 := st.MakePod().Name("pg1-pod2").Namespace("default").UID("pg1-pod2").PodGroupName("pg1").Obj()
	pg1Pod3 := st.MakePod().Name("pg1-pod3").Namespace("default").UID("pg1-pod3").PodGroupName("pg1").Obj()
	pg2Pod1 := st.MakePod().Name("pg2-pod1").Namespace("default").UID("pg2-pod1").PodGroupName("pg2").Obj()

	childCompositePodGroup1 := st.MakeCompositePodGroup().Name("cpg1").Namespace("default").ParentCompositePodGroup("cpg").Obj()
	childCompositePodGroup2 := st.MakeCompositePodGroup().Name("cpg2").Namespace("default").ParentCompositePodGroup("cpg").Obj()
	childPodGroup3 := st.MakePodGroup().Name("pg3").Namespace("default").ParentCompositePodGroup("cpg1").MinCount(1).Obj()
	childPodGroup4 := st.MakePodGroup().Name("pg4").Namespace("default").ParentCompositePodGroup("cpg1").MinCount(1).Obj()
	childPodGroup5 := st.MakePodGroup().Name("pg5").Namespace("default").ParentCompositePodGroup("cpg2").MinCount(1).Obj()
	pg3Pod1 := st.MakePod().Name("pg3-pod1").Namespace("default").UID("pg3-pod1").PodGroupName("pg3").Obj()
	pg4Pod1 := st.MakePod().Name("pg4-pod1").Namespace("default").UID("pg4-pod1").PodGroupName("pg4").Obj()
	pg5Pod1 := st.MakePod().Name("pg5-pod1").Namespace("default").UID("pg5-pod1").PodGroupName("pg5").Obj()

	rootCompositePodGroupKey := fwk.CompositePodGroupKey("default", "cpg")
	childPodGroup1Key := fwk.PodGroupKey("default", "pg1")
	childCompositePodGroup1Key := fwk.CompositePodGroupKey("default", "cpg1")

	tests := []struct {
		name                      string
		root                      *fwk.GenericPodGroup
		cpgs                      []*schedulingv1beta1.CompositePodGroup
		pgs                       []*schedulingv1beta1.PodGroup
		pods                      []*v1.Pod
		placementFeasibleStatuses [][]fwk.Code
		assumeErrorAt             string
		placements                map[fwk.EntityKey]map[string][]string
		placementScores           map[fwk.EntityKey]map[string]int64
		wantStatus                fwk.Code
		wantAssumeEvents          []string
		wantChildStatus           map[fwk.EntityKey]fwk.Code
	}{
		{
			name: "Root pod group is not assumed",
			root: fwk.NewGenericPodGroup(rootPodGroup),
			pgs:  []*schedulingv1beta1.PodGroup{rootPodGroup},
			pods: []*v1.Pod{podGroupPod},
			wantAssumeEvents: []string{
				"assume pod@node1",
				"revert pod@node1",
			},
		},
		{
			name: "Pod group child with subsequent siblings within hierarchy is assumed, last child and root are not",
			root: fwk.NewGenericCompositePodGroup(rootCompositePodGroup),
			cpgs: []*schedulingv1beta1.CompositePodGroup{rootCompositePodGroup},
			pgs:  []*schedulingv1beta1.PodGroup{childPodGroup1, childPodGroup2},
			pods: []*v1.Pod{pg1Pod1, pg2Pod1},
			wantAssumeEvents: []string{
				"assume pg1-pod1@node1",
				"revert pg1-pod1@node1",
				"assume pg1-pod1@node1",
				"assume pg2-pod1@node1",
				"revert pg2-pod1@node1",
				"revert pg1-pod1@node1",
			},
		},
		{
			name: "Multi-pod child pod group followed by a sibling assumes and reverts pods in LIFO order",
			root: fwk.NewGenericCompositePodGroup(rootCompositePodGroup),
			cpgs: []*schedulingv1beta1.CompositePodGroup{rootCompositePodGroup},
			pgs:  []*schedulingv1beta1.PodGroup{childPodGroup1, childPodGroup2},
			pods: []*v1.Pod{pg1Pod1, pg1Pod2, pg1Pod3, pg2Pod1},
			wantAssumeEvents: []string{
				"assume pg1-pod1@node1",
				"assume pg1-pod2@node1",
				"assume pg1-pod3@node1",
				"revert pg1-pod3@node1",
				"revert pg1-pod2@node1",
				"revert pg1-pod1@node1",
				"assume pg1-pod1@node1",
				"assume pg1-pod2@node1",
				"assume pg1-pod3@node1",
				"assume pg2-pod1@node1",
				"revert pg2-pod1@node1",
				"revert pg1-pod3@node1",
				"revert pg1-pod2@node1",
				"revert pg1-pod1@node1",
			},
		},
		{
			name: "Basic child pod group with a subsequent sibling skips subtree assume when it had no scheduled pods",
			root: fwk.NewGenericCompositePodGroup(rootCompositePodGroup),
			cpgs: []*schedulingv1beta1.CompositePodGroup{rootCompositePodGroup},
			pgs:  []*schedulingv1beta1.PodGroup{childBasicPodGroup1, childPodGroup2},
			pods: []*v1.Pod{pg2Pod1},
			wantAssumeEvents: []string{
				"assume pg2-pod1@node1",
				"revert pg2-pod1@node1",
			},
		},
		{
			name: "Composite pod group child with subsequent siblings is assumed, last child and root are not",
			root: fwk.NewGenericCompositePodGroup(rootCompositePodGroup),
			cpgs: []*schedulingv1beta1.CompositePodGroup{rootCompositePodGroup, childCompositePodGroup1, childCompositePodGroup2},
			pgs:  []*schedulingv1beta1.PodGroup{childPodGroup3, childPodGroup4, childPodGroup5},
			pods: []*v1.Pod{pg3Pod1, pg4Pod1, pg5Pod1},
			wantAssumeEvents: []string{
				"assume pg3-pod1@node1",
				"revert pg3-pod1@node1",
				"assume pg3-pod1@node1",
				"assume pg4-pod1@node1",
				"revert pg4-pod1@node1",
				"revert pg3-pod1@node1",
				"assume pg3-pod1@node1",
				"assume pg4-pod1@node1",
				"assume pg5-pod1@node1",
				"revert pg5-pod1@node1",
				"revert pg4-pod1@node1",
				"revert pg3-pod1@node1",
			},
		},
		{
			name:                      "Child is not assumed when PlacementFeasible rejects the parent",
			root:                      fwk.NewGenericCompositePodGroup(rootCompositePodGroup),
			cpgs:                      []*schedulingv1beta1.CompositePodGroup{rootCompositePodGroup},
			pgs:                       []*schedulingv1beta1.PodGroup{childPodGroup1, childPodGroup2},
			pods:                      []*v1.Pod{pg1Pod1, pg2Pod1},
			placementFeasibleStatuses: [][]fwk.Code{{fwk.Success, fwk.Unschedulable}},
			wantStatus:                fwk.Unschedulable,
			wantAssumeEvents: []string{
				"assume pg1-pod1@node1",
				"revert pg1-pod1@node1",
			},
		},
		{
			name: "Child is assumed on its best-scored placement, not the last evaluated one",
			root: fwk.NewGenericCompositePodGroup(rootCompositePodGroup),
			cpgs: []*schedulingv1beta1.CompositePodGroup{rootCompositePodGroup},
			pgs:  []*schedulingv1beta1.PodGroup{childPodGroup1, childPodGroup2},
			pods: []*v1.Pod{pg1Pod1, pg2Pod1},
			placements: map[fwk.EntityKey]map[string][]string{
				rootCompositePodGroupKey: {"all": {"node1", "node2"}},
				childPodGroup1Key:        {"on-node1": {"node1"}, "on-node2": {"node2"}},
			},
			placementScores: map[fwk.EntityKey]map[string]int64{
				childPodGroup1Key: {"on-node1": 100, "on-node2": 1},
			},
			wantAssumeEvents: []string{
				"assume pg1-pod1@node1",
				"revert pg1-pod1@node1",
				"assume pg1-pod1@node2",
				"revert pg1-pod1@node2",
				"assume pg1-pod1@node1",
				"assume pg2-pod1@node1",
				"revert pg2-pod1@node1",
				"revert pg1-pod1@node1",
			},
		},
		{
			name:          "Assume failure is attributed to the podgroup child",
			root:          fwk.NewGenericCompositePodGroup(rootCompositePodGroup),
			cpgs:          []*schedulingv1beta1.CompositePodGroup{rootCompositePodGroup},
			pgs:           []*schedulingv1beta1.PodGroup{childPodGroup1, childPodGroup2},
			pods:          []*v1.Pod{pg1Pod1, pg2Pod1},
			assumeErrorAt: "pg1-pod1@node1",
			wantStatus:    fwk.Error,
			wantAssumeEvents: []string{
				"assume pg1-pod1@node1",
				"revert pg1-pod1@node1",
				"assume pg1-pod1@node1",
				"revert pg1-pod1@node1",
			},
			wantChildStatus: map[fwk.EntityKey]fwk.Code{childPodGroup1Key: fwk.Error},
		},
		{
			name:          "Assume failure is attributed to the composite pod group child",
			root:          fwk.NewGenericCompositePodGroup(rootCompositePodGroup),
			cpgs:          []*schedulingv1beta1.CompositePodGroup{rootCompositePodGroup, childCompositePodGroup1, childCompositePodGroup2},
			pgs:           []*schedulingv1beta1.PodGroup{childPodGroup3, childPodGroup4, childPodGroup5},
			pods:          []*v1.Pod{pg3Pod1, pg4Pod1, pg5Pod1},
			assumeErrorAt: "pg4-pod1@node1",
			wantStatus:    fwk.Error,
			wantAssumeEvents: []string{
				"assume pg3-pod1@node1",
				"revert pg3-pod1@node1",
				"assume pg3-pod1@node1",
				"assume pg4-pod1@node1",
				"revert pg4-pod1@node1",
				"revert pg3-pod1@node1",
				"assume pg3-pod1@node1",
				"assume pg4-pod1@node1",
				"revert pg4-pod1@node1",
				"revert pg3-pod1@node1",
			},
			wantChildStatus: map[fwk.EntityKey]fwk.Code{childCompositePodGroup1Key: fwk.Error},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			var root *framework.QueuedPodGroupInfo
			if rootCPG := tt.root.GetCompositePodGroup(); rootCPG != nil {
				root = buildHierarchicalQueuedPodGroupInfo(rootCPG, tt.cpgs, tt.pgs, tt.pods)
			} else {
				root = newQueuedPodGroupInfo(&framework.PodGroupInfo{GenericPodGroup: tt.root}, &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: tt.pods[0]}})
			}
			nodes := []*v1.Node{st.MakeNode().Name("node1").Obj(), st.MakeNode().Name("node2").Obj()}
			snapshot := internalcache.NewTestSnapshotWithPodGroups(nil, nodes, tt.pgs, tt.cpgs)

			placements := map[fwk.EntityKey]map[string][]string{}
			for _, pg := range tt.pgs {
				placements[fwk.PodGroupKey(pg.Namespace, pg.Name)] = map[string][]string{"placement1": {"node1"}}
			}
			for _, cpg := range tt.cpgs {
				placements[fwk.CompositePodGroupKey(cpg.Namespace, cpg.Name)] = map[string][]string{"placement1": {"node1"}}
			}
			maps.Copy(placements, tt.placements)
			placementPlugin := &fakePlacementPlugin{
				name:                     "FakeGeneratorPlugin",
				generatePlacementsResult: placements,
				scorePlacementsResult:    tt.placementScores,
			}
			orderedPlacementGeneratePlugin := &orderedPlacementPlugin{placementPlugin}
			placementFeasiblePlugin := &fakePlacementFeasiblePlugin{placementFeasibleStatuses: tt.placementFeasibleStatuses}
			assumeAndReservePlugin := &fakeAssumeAndReserveRecorderPlugin{assumeErrorAt: tt.assumeErrorAt}

			queue := internalqueue.NewTestQueue(ctx, nil)
			schedFwk, err := tf.NewFramework(ctx,
				[]tf.RegisterPluginFunc{
					tf.RegisterPlacementGeneratePlugin(orderedPlacementGeneratePlugin.Name(), func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
						return orderedPlacementGeneratePlugin, nil
					}),
					tf.RegisterPlacementScorePlugin(placementPlugin.Name(), func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
						return placementPlugin, nil
					}, 1),
					tf.RegisterPlacementFeasiblePlugin(placementFeasiblePlugin.Name(), func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
						return placementFeasiblePlugin, nil
					}),
					tf.RegisterReservePlugin(assumeAndReservePlugin.Name(), func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
						return assumeAndReservePlugin, nil
					}),
					tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
					tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
				},
				"test-scheduler",
				frameworkruntime.WithSnapshotSharedLister(snapshot),
				frameworkruntime.WithPodNominator(queue),
			)
			if err != nil {
				t.Fatalf("Failed to create new framework: %v", err)
			}

			cache := internalcache.New(ctx, nil, true, true)
			podGroupAlgorithm := testPodGroupAlgorithm(t, ctx, snapshot, cache, queue)

			results := podGroupAlgorithm.RunRootSchedulingAlgorithm(ctx, schedFwk, framework.NewCycleState(), root)

			if got := results[root.GetKey()].status.Code(); got != tt.wantStatus {
				t.Fatalf("Unexpected root status code: got %v, want %v (status: %v)", got, tt.wantStatus, results[root.GetKey()].status)
			}
			if diff := cmp.Diff(tt.wantAssumeEvents, assumeAndReservePlugin.events); diff != "" {
				t.Errorf("Unexpected assume events (-want, +got):\n%s", diff)
			}
			for key, wantStatus := range tt.wantChildStatus {
				childRes, ok := results[key]
				if !ok {
					t.Fatalf("Missing result for child %v", key)
				}
				if got := childRes.status.Code(); got != wantStatus {
					t.Errorf("Unexpected status code for child %v: got %v, want %v", key, got, wantStatus)
				}
			}
		})
	}
}
