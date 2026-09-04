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
	"strings"
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
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/queuesort"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	tf "k8s.io/kubernetes/pkg/scheduler/testing/framework"
)

func testPodGroupAlgorithm(snapshot *internalcache.Snapshot, cache internalcache.Cache, queue internalqueue.SchedulingQueue) *PodGroupSchedulingAlgorithm {
	return NewPodGroupAlgorithm(snapshot, NewSchedulingAlgorithm(snapshot, cache, WithCurrentCycleProvider(queue.SchedulingCycle)))
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
		QueuedPodInfos: map[fwk.EntityKey][]*framework.QueuedPodInfo{fwk.PodGroupKey("default", "pg"): queuedPodInfos},
		PodGroupInfo: &framework.PodGroupInfo{
			GenericPodGroup: framework.NewGenericPodGroup(testPodGroup),
			UnscheduledPods: []*v1.Pod{p1, p2, p3},
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
					cache.AddGenericPodGroup(framework.NewGenericPodGroup(testPodGroup))

					if err := cache.UpdateSnapshot(logger, snapshot); err != nil {
						t.Fatalf("Failed to update snapshot: %v", err)
					}

					podGroupAlgorithm := testPodGroupAlgorithm(snapshot, cache, queue)
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
	podGroupPod := st.MakePod().Name("foo").UID("foo").PodGroupName("pg").Obj()
	testPodGroup := &schedulingv1beta1.PodGroup{
		ObjectMeta: metav1.ObjectMeta{Name: "pg", Namespace: "default"},
	}

	podInfo, err := framework.NewPodInfo(st.MakePod().Name("foo").UID("foo").PodGroupName("pg").Obj())
	if err != nil {
		t.Fatalf("Failed to create pod info: %v", err)
	}
	podGroupPodInfo := &framework.QueuedPodInfo{PodInfo: podInfo}

	pgKeyVal := fwk.PodGroupKey("default", "pg")
	queuedPodInfos := []*framework.QueuedPodInfo{{PodInfo: &framework.PodInfo{Pod: podGroupPod}}}
	pgInfo := &framework.QueuedPodGroupInfo{
		QueuedPodInfos: map[fwk.EntityKey][]*framework.QueuedPodInfo{pgKeyVal: queuedPodInfos},
		PodGroupInfo: &framework.PodGroupInfo{
			GenericPodGroup: framework.NewGenericPodGroup(testPodGroup),
			UnscheduledPods: []*v1.Pod{podGroupPod},
		},
	}

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
				status: fwk.NewStatus(fwk.Error, "running PlacementScore plugins: plugin \"FakePlacementPlugin\" failed with: error for test").WithPlugin("FakePlacementPlugin"),
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
				cache.AddGenericPodGroup(framework.NewGenericPodGroup(testPodGroup))

				if err := cache.UpdateSnapshot(logger, snapshot); err != nil {
					t.Fatalf("Failed to update snapshot: %v", err)
				}

				metrics.GeneratedPlacementsTotal.Reset()
				metrics.PlacementEvaluations.Reset()
				metrics.PlacementEvaluationDuration.Reset()

				podGroupAlgorithm := testPodGroupAlgorithm(snapshot, cache, queue)
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

				pgKeyVal := fwk.PodGroupKey("default", "pg")
				queuedPodInfos := []*framework.QueuedPodInfo{{PodInfo: &framework.PodInfo{Pod: podGroupPod}}}
				testPodGroup := st.MakePodGroup().Namespace("default").Name("pg").Obj()
				pgInfo := &framework.QueuedPodGroupInfo{
					QueuedPodInfos: map[fwk.EntityKey][]*framework.QueuedPodInfo{pgKeyVal: queuedPodInfos},
					PodGroupInfo: &framework.PodGroupInfo{
						GenericPodGroup: framework.NewGenericPodGroup(testPodGroup),
						UnscheduledPods: []*v1.Pod{podGroupPod},
					},
				}

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
				cache.AddGenericPodGroup(framework.NewGenericPodGroup(testPodGroup))

				if err := cache.UpdateSnapshot(logger, snapshot); err != nil {
					t.Fatalf("Failed to update snapshot: %v", err)
				}

				podGroupAlgorithm := testPodGroupAlgorithm(snapshot, cache, queue)
				result, _ := podGroupAlgorithm.podGroupSchedulingPlacementAlgorithm(ctx, schedFwk, framework.NewCycleState(), pgInfo.PodGroupInfo, pgInfo)

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
			cache.AddGenericPodGroup(framework.NewGenericPodGroup(testPodGroup))

			if err := cache.UpdateSnapshot(logger, snapshot); err != nil {
				t.Fatalf("Failed to update snapshot: %v", err)
			}

			queuedPodInfos := []*framework.QueuedPodInfo{{PodInfo: &framework.PodInfo{Pod: podGroupPod}}}
			pgInfo := &framework.QueuedPodGroupInfo{
				QueuedPodInfos: map[fwk.EntityKey][]*framework.QueuedPodInfo{fwk.PodGroupKey("default", "pg"): queuedPodInfos},
				PodGroupInfo: &framework.PodGroupInfo{
					GenericPodGroup: framework.NewGenericPodGroup(testPodGroup),
					UnscheduledPods: []*v1.Pod{podGroupPod},
				},
			}
			podGroupAlgorithm := testPodGroupAlgorithm(snapshot, cache, queue)
			result, _ := podGroupAlgorithm.podGroupSchedulingPlacementAlgorithm(ctx, schedFwk, framework.NewCycleState(), pgInfo.PodGroupInfo, pgInfo)
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

	leafPGInfo := &framework.PodGroupInfo{GenericPodGroup: framework.NewGenericPodGroup(pg), UnscheduledPods: []*v1.Pod{p1}}
	midPGInfo := &framework.PodGroupInfo{GenericPodGroup: framework.NewGenericCompositePodGroup(midcpg), Children: []*framework.PodGroupInfo{leafPGInfo}}
	rootPGInfo := &framework.PodGroupInfo{GenericPodGroup: framework.NewGenericCompositePodGroup(rootcpg), Children: []*framework.PodGroupInfo{midPGInfo}}

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
	cache.AddGenericPodGroup(framework.NewGenericCompositePodGroup(rootcpg))
	cache.AddGenericPodGroup(framework.NewGenericCompositePodGroup(midcpg))
	cache.AddGenericPodGroup(framework.NewGenericPodGroup(pg))

	if err := cache.UpdateSnapshot(logger, snapshot); err != nil {
		t.Fatalf("Failed to update snapshot: %v", err)
	}

	cpgQueuedInfo := &framework.QueuedPodGroupInfo{
		QueuedPodInfos: map[fwk.EntityKey][]*framework.QueuedPodInfo{
			leafPGInfo.GetKey(): {queuedPodInfo1},
		},
		PodGroupInfo: rootPGInfo,
	}

	podGroupAlgorithm := testPodGroupAlgorithm(snapshot, cache, queue)
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

	childPGInfo1 := &framework.PodGroupInfo{GenericPodGroup: framework.NewGenericPodGroup(pg1), UnscheduledPods: []*v1.Pod{p1}}
	childPGInfo2 := &framework.PodGroupInfo{GenericPodGroup: framework.NewGenericPodGroup(pg2), UnscheduledPods: []*v1.Pod{p2}}

	rootPGInfo := &framework.PodGroupInfo{GenericPodGroup: framework.NewGenericCompositePodGroup(cpg), Children: []*framework.PodGroupInfo{childPGInfo1, childPGInfo2}}

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
		placementPlugin           fakePlacementPlugin
		placementFeasibleStatuses [][]fwk.Code
		expectedResults           map[fwk.EntityKey]PodGroupScheduledResult
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
					status: fwk.NewStatus(fwk.Error, "composite pod group evaluation failed due to child error: injected error"),
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
			cache.AddGenericPodGroup(framework.NewGenericCompositePodGroup(cpg))
			cache.AddGenericPodGroup(framework.NewGenericPodGroup(pg1))
			cache.AddGenericPodGroup(framework.NewGenericPodGroup(pg2))
			cache.AddPodGroupMember(p1)
			cache.AddPodGroupMember(p2)

			if err := cache.UpdateSnapshot(logger, snapshot); err != nil {
				t.Fatalf("Failed to update snapshot: %v", err)
			}

			cpgInfo := &framework.QueuedPodGroupInfo{
				QueuedPodInfos: map[fwk.EntityKey][]*framework.QueuedPodInfo{
					childPGInfo1.GetKey(): {queuedPodInfo1},
					childPGInfo2.GetKey(): {queuedPodInfo2},
				},
				PodGroupInfo: rootPGInfo,
			}

			podGroupAlgorithm := testPodGroupAlgorithm(snapshot, cache, queue)
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

	childPGInfo1 := &framework.PodGroupInfo{GenericPodGroup: framework.NewGenericPodGroup(pg1), UnscheduledPods: []*v1.Pod{p1}}
	childPGInfo2 := &framework.PodGroupInfo{GenericPodGroup: framework.NewGenericPodGroup(pg2), UnscheduledPods: []*v1.Pod{p2}}

	rootPGInfo := &framework.PodGroupInfo{GenericPodGroup: framework.NewGenericCompositePodGroup(cpg), Children: []*framework.PodGroupInfo{childPGInfo1, childPGInfo2}}

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
				tf.RegisterPreEnqueuePlugin(gangscheduling.Name, gangPluginFactory),
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
			cache.AddGenericPodGroup(framework.NewGenericCompositePodGroup(cpg))
			cache.AddGenericPodGroup(framework.NewGenericPodGroup(pg1))
			cache.AddGenericPodGroup(framework.NewGenericPodGroup(pg2))
			cache.AddPodGroupMember(p1)
			cache.AddPodGroupMember(p2)

			if err := cache.UpdateSnapshot(logger, snapshot); err != nil {
				t.Fatalf("Failed to update snapshot: %v", err)
			}

			cpgInfo := &framework.QueuedPodGroupInfo{
				QueuedPodInfos: map[fwk.EntityKey][]*framework.QueuedPodInfo{
					childPGInfo1.GetKey(): {queuedPodInfo1},
					childPGInfo2.GetKey(): {queuedPodInfo2},
				},
				PodGroupInfo: rootPGInfo,
			}

			podGroupAlgorithm := testPodGroupAlgorithm(snapshot, cache, queue)
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
				pgi := &framework.PodGroupInfo{Children: children}
				if len(children) > 0 {
					cpg := st.MakeCompositePodGroup().Name(node.name).Namespace("default").Obj()
					pgi.GenericPodGroup = framework.
						NewGenericCompositePodGroup(cpg)

				} else {
					pg := st.MakePodGroup().Name(node.name).Namespace("default").Obj()
					pgi.GenericPodGroup = framework.
						NewGenericPodGroup(pg)
				}
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

			podGroupAlgorithm := testPodGroupAlgorithm(snapshot, nil, queue)
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
