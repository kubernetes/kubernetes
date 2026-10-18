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

package preemption

import (
	"context"
	"fmt"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
	compresource "k8s.io/component-helpers/resource"
	configv1 "k8s.io/kube-scheduler/config/v1"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler"
	configtesting "k8s.io/kubernetes/pkg/scheduler/apis/config/testing"
	schedmetrics "k8s.io/kubernetes/pkg/scheduler/metrics"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/kubernetes/test/integration/scheduler/preemption/asyncframework"
	testutils "k8s.io/kubernetes/test/integration/util"
	"k8s.io/utils/ptr"
)

func TestDeferredResizePodPreemption(t *testing.T) {
	// Setup API server with feature gates enabled
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.InPlacePodVerticalScaling:                    true,
		features.InPlacePodVerticalScalingSchedulerPreemption: true,
		features.DRANodeAllocatableResources:                  true,
	})
	registerResizeMetrics()

	cfg := configtesting.V1ToInternalWithDefaults(t, configv1.KubeSchedulerConfiguration{
		Profiles: []configv1.KubeSchedulerProfile{{
			SchedulerName: new(v1.DefaultSchedulerName),
		}},
	})

	tests := []struct {
		name                                  string
		nodeCapacityCPU                       string
		nodeCapacityMem                       string
		nodeCapacityPods                      string
		nodePreemptionPolicy                  *v1.NodePodPreemptionPolicy
		existingPods                          []*v1.Pod
		existingPodAllocatedCPU               map[string]string
		terminatingByPreemptionPods           map[string]bool
		preemptorConfig                       *testutils.PausePodConfig
		preemptorNodeAllocatableClaimStatuses []v1.NodeAllocatableResourceClaimStatus
		expectEvictedNames                    []string
		resizePodWhileVictimTerminating       string
		resizePodNewCPU                       string
		expectEvaluationCompletedImmediately  bool
		expectFailedSchedulingEvent           bool
	}{
		{
			name:            "preempt single low priority pod",
			nodeCapacityCPU: "300m",
			nodeCapacityMem: "300",
			existingPods: []*v1.Pod{
				initPausePod(&testutils.PausePodConfig{
					Name:     "victim-1",
					Priority: &asyncframework.LowPriority,
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(100, resource.DecimalSI)},
					},
				}),
			},
			preemptorConfig: &testutils.PausePodConfig{
				Name:     "preemptor-pod",
				Priority: &asyncframework.HighPriority,
				Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
					v1.ResourceCPU:    *resource.NewMilliQuantity(300, resource.DecimalSI),
					v1.ResourceMemory: *resource.NewQuantity(100, resource.DecimalSI)},
				},
			},
			expectEvictedNames: []string{"victim-1"},
		},
		{
			name:            "preempt multiple low priority pods",
			nodeCapacityCPU: "300m",
			nodeCapacityMem: "300",
			existingPods: []*v1.Pod{
				initPausePod(&testutils.PausePodConfig{
					Name:     "victim-1",
					Priority: &asyncframework.LowPriority,
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(100, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(50, resource.DecimalSI)},
					},
				}),
				initPausePod(&testutils.PausePodConfig{
					Name:     "victim-2",
					Priority: &asyncframework.LowPriority,
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(100, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(50, resource.DecimalSI)},
					},
				}),
			},
			preemptorConfig: &testutils.PausePodConfig{
				Name:     "preemptor-pod",
				Priority: &asyncframework.HighPriority,
				Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
					v1.ResourceCPU:    *resource.NewMilliQuantity(300, resource.DecimalSI),
					v1.ResourceMemory: *resource.NewQuantity(100, resource.DecimalSI)},
				},
			},
			expectEvictedNames: []string{"victim-1", "victim-2"},
		},
		{
			name:             "preempt single low priority pod when node is at max pod capacity",
			nodeCapacityCPU:  "300m",
			nodeCapacityMem:  "300",
			nodeCapacityPods: "2",
			existingPods: []*v1.Pod{
				initPausePod(&testutils.PausePodConfig{
					Name:     "victim-1",
					Priority: &asyncframework.LowPriority,
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(100, resource.DecimalSI)},
					},
				}),
			},
			preemptorConfig: &testutils.PausePodConfig{
				Name:     "preemptor-pod",
				Priority: &asyncframework.HighPriority,
				Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
					v1.ResourceCPU:    *resource.NewMilliQuantity(300, resource.DecimalSI),
					v1.ResourceMemory: *resource.NewQuantity(100, resource.DecimalSI)},
				},
			},
			expectEvictedNames: []string{"victim-1"},
		},
		{
			name:            "no preemption when preemptor policy is PreemptNever",
			nodeCapacityCPU: "300m",
			nodeCapacityMem: "300",
			existingPods: []*v1.Pod{
				initPausePod(&testutils.PausePodConfig{
					Name:     "victim-1",
					Priority: &asyncframework.LowPriority,
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(100, resource.DecimalSI)},
					},
				}),
			},
			preemptorConfig: &testutils.PausePodConfig{
				Name:             "preemptor-pod",
				Priority:         &asyncframework.HighPriority,
				PreemptionPolicy: new(v1.PreemptNever),
				Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
					v1.ResourceCPU:    *resource.NewMilliQuantity(300, resource.DecimalSI),
					v1.ResourceMemory: *resource.NewQuantity(100, resource.DecimalSI)},
				},
			},
			expectEvictedNames:          nil,
			expectFailedSchedulingEvent: true,
		},
		{
			name:            "sets SchedulerEvaluationCompleted=True and leaves queue when deferred resize fits immediately",
			nodeCapacityCPU: "300m",
			nodeCapacityMem: "300",
			existingPods:    nil, // fits immediately
			preemptorConfig: &testutils.PausePodConfig{
				Name:     "preemptor-pod",
				Priority: &asyncframework.HighPriority,
				Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
					v1.ResourceCPU:    *resource.NewMilliQuantity(300, resource.DecimalSI),
					v1.ResourceMemory: *resource.NewQuantity(100, resource.DecimalSI)},
				},
			},
			expectEvictedNames:                   nil,
			expectEvaluationCompletedImmediately: true,
		},
		{
			name:            "node-level preemption policy disables preemption",
			nodeCapacityCPU: "300m",
			nodeCapacityMem: "300",
			nodePreemptionPolicy: &v1.NodePodPreemptionPolicy{
				DisableResizePreemption: []string{"test-operator"},
			},
			existingPods: []*v1.Pod{
				initPausePod(&testutils.PausePodConfig{
					Name:     "victim-1",
					Priority: &asyncframework.LowPriority,
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(100, resource.DecimalSI)},
					},
				}),
			},
			preemptorConfig: &testutils.PausePodConfig{
				Name:     "preemptor-pod",
				Priority: &asyncframework.HighPriority,
				Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
					v1.ResourceCPU:    *resource.NewMilliQuantity(300, resource.DecimalSI),
					v1.ResourceMemory: *resource.NewQuantity(100, resource.DecimalSI)},
				},
			},
			expectEvictedNames:          nil,
			expectFailedSchedulingEvent: true,
		},
		{
			name:            "reprieve with multiple deferred resize victims uses max(allocated, actual) and preempts only 1 victim",
			nodeCapacityCPU: "300m",
			nodeCapacityMem: "300",
			existingPods: []*v1.Pod{
				initPausePod(&testutils.PausePodConfig{
					Name:     "victim-1",
					Priority: &asyncframework.MediumPriority,
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(50, resource.DecimalSI)},
					},
				}),
				initPausePod(&testutils.PausePodConfig{
					Name:     "victim-2",
					Priority: &asyncframework.LowPriority,
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(50, resource.DecimalSI)},
					},
				}),
			},
			// Both victims have unfulfilled desired requests of 200m, but only 100m allocated.
			// Preemptor wants 200m on a 300m node, so higher-priority victim-1 (100m allocated)
			// is reprieved and only victim-2 needs to be preempted.
			existingPodAllocatedCPU: map[string]string{
				"victim-1": "100m",
				"victim-2": "100m",
			},
			preemptorConfig: &testutils.PausePodConfig{
				Name:     "preemptor-pod",
				Priority: &asyncframework.HighPriority,
				Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
					v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
					v1.ResourceMemory: *resource.NewQuantity(100, resource.DecimalSI)},
				},
			},
			expectEvictedNames: []string{"victim-2"},
		},
		{
			name:            "reprieve with multiple deferred resize victims reprieves all victims (0 victims evicted) and sets SchedulerEvaluationCompleted=True",
			nodeCapacityCPU: "300m",
			nodeCapacityMem: "300",
			existingPods: []*v1.Pod{
				initPausePod(&testutils.PausePodConfig{
					Name:     "victim-1",
					Priority: &asyncframework.LowPriority,
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(50, resource.DecimalSI)},
					},
				}),
				initPausePod(&testutils.PausePodConfig{
					Name:     "victim-2",
					Priority: &asyncframework.LowPriority,
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(50, resource.DecimalSI)},
					},
				}),
			},
			// Both victims have unfulfilled desired requests of 200m, but only 50m allocated.
			// Preemptor wants 200m on a 300m node (200m + 50m + 50m = 300m), so both victims
			// are reprieved and 0 victims are evicted.
			existingPodAllocatedCPU: map[string]string{
				"victim-1": "50m",
				"victim-2": "50m",
			},
			preemptorConfig: &testutils.PausePodConfig{
				Name:     "preemptor-pod",
				Priority: &asyncframework.HighPriority,
				Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
					v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
					v1.ResourceMemory: *resource.NewQuantity(100, resource.DecimalSI)},
				},
			},
			expectEvictedNames:                   nil,
			expectEvaluationCompletedImmediately: true,
		},
		{
			name:            "DRA node-allocatable resource claim statuses accounted for during deferred resize preemption",
			nodeCapacityCPU: "300m",
			nodeCapacityMem: "300",
			existingPods: []*v1.Pod{
				initPausePod(&testutils.PausePodConfig{
					Name:     "victim-1",
					Priority: &asyncframework.LowPriority,
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(150, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(50, resource.DecimalSI)},
					},
				}),
			},
			preemptorConfig: &testutils.PausePodConfig{
				Name:     "preemptor-pod",
				Priority: &asyncframework.HighPriority,
				Resources: &v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(150, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(100, resource.DecimalSI),
					},
					Claims: []v1.ResourceClaim{{Name: "dra-cpu-claim"}},
				},
			},
			preemptorNodeAllocatableClaimStatuses: []v1.NodeAllocatableResourceClaimStatus{
				{
					ResourceClaimName: "dra-cpu-claim",
					Containers:        []string{"preemptor-pod"},
					Mapping: []v1.NodeAllocatableMappedResources{{
						Name:     v1.ResourceCPU,
						Quantity: new(resource.MustParse("100m")),
					}},
				},
			},
			expectEvictedNames: []string{"victim-1"},
		},
		{
			name:            "terminating victim from preemption suppresses double preemption of second victim",
			nodeCapacityCPU: "400m",
			nodeCapacityMem: "400",
			existingPods: []*v1.Pod{
				initPausePod(&testutils.PausePodConfig{
					Name:     "victim-1",
					Priority: &asyncframework.LowPriority,
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(100, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(50, resource.DecimalSI)},
					},
				}),
				initPausePod(&testutils.PausePodConfig{
					Name:     "victim-2",
					Priority: &asyncframework.MediumPriority,
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(100, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(50, resource.DecimalSI)},
					},
				}),
				initPausePod(&testutils.PausePodConfig{
					Name:     "other-high-pod",
					Priority: &asyncframework.HighPriority,
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(100, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(50, resource.DecimalSI)},
					},
				}),
			},
			preemptorConfig: &testutils.PausePodConfig{
				Name:     "preemptor-pod",
				Priority: &asyncframework.HighPriority,
				Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
					v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
					v1.ResourceMemory: *resource.NewQuantity(100, resource.DecimalSI)},
				},
			},
			expectEvictedNames:              []string{"victim-1"},
			resizePodWhileVictimTerminating: "other-high-pod",
			resizePodNewCPU:                 "80m",
		},
		{
			name:            "pre-existing terminating preemption victim on node suppresses preemption of second victim",
			nodeCapacityCPU: "300m",
			nodeCapacityMem: "300",
			existingPods: []*v1.Pod{
				initPausePod(&testutils.PausePodConfig{
					Name:     "victim-1",
					Priority: &asyncframework.LowPriority,
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(100, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(50, resource.DecimalSI)},
					},
				}),
				initPausePod(&testutils.PausePodConfig{
					Name:     "victim-2",
					Priority: &asyncframework.MediumPriority,
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(100, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(50, resource.DecimalSI)},
					},
				}),
			},
			terminatingByPreemptionPods: map[string]bool{
				"victim-1": true,
			},
			preemptorConfig: &testutils.PausePodConfig{
				Name:     "preemptor-pod",
				Priority: &asyncframework.HighPriority,
				Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
					v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
					v1.ResourceMemory: *resource.NewQuantity(100, resource.DecimalSI)},
				},
			},
			expectEvictedNames:          nil,
			expectFailedSchedulingEvent: true,
		},
	}

	// Initialize API server and Scheduler ONCE for all table cases
	testCtx := testutils.InitTestSchedulerWithOptions(t,
		testutils.InitTestAPIServer(t, "def-preempt", nil),
		0,
		scheduler.WithProfiles(cfg.Profiles...),
	)
	defer testCtx.SchedulerCloseFn()
	testutils.SyncSchedulerInformerFactory(testCtx)
	go testCtx.Scheduler.Run(testCtx.Ctx)

	cs := testCtx.ClientSet

	for idx, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			nodeName := fmt.Sprintf("preempt-node-%d", idx)

			// Create node
			podCap := "32"
			if tt.nodeCapacityPods != "" {
				podCap = tt.nodeCapacityPods
			}
			nodeRes := map[v1.ResourceName]string{
				v1.ResourcePods:   podCap,
				v1.ResourceCPU:    tt.nodeCapacityCPU,
				v1.ResourceMemory: tt.nodeCapacityMem,
			}
			nodeObject := st.MakeNode().Name(nodeName).Capacity(nodeRes).Label("node", nodeName).Obj()
			if tt.nodePreemptionPolicy != nil {
				nodeObject.Spec.PodPreemptionPolicy = tt.nodePreemptionPolicy
			}
			if _, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, nodeObject, metav1.CreateOptions{}); err != nil {
				t.Fatalf("Failed to create node: %v", err)
			}
			t.Cleanup(func() {
				if err := cs.CoreV1().Nodes().Delete(testCtx.Ctx, nodeName, metav1.DeleteOptions{}); err != nil && !apierrors.IsNotFound(err) {
					t.Errorf("Failed to delete node %s: %v", nodeName, err)
				}
			})

			// Create and run existing pods (if any) on nodeName
			var pods []*v1.Pod
			t.Cleanup(func() {
				testutils.CleanupPods(testCtx.Ctx, cs, t, pods)
			})
			for _, p := range tt.existingPods {
				podCopy := p.DeepCopy()
				podCopy.Namespace = testCtx.NS.Name
				podCopy.Spec.NodeName = nodeName
				runningPod, err := runPausePod(cs, podCopy)
				if err != nil {
					t.Fatalf("Failed running pause pod %v: %v", podCopy.Name, err)
				}
				if allocCPU, ok := tt.existingPodAllocatedCPU[podCopy.Name]; ok {
					runningPod.Status.Phase = v1.PodRunning
					runningPod.Status.ContainerStatuses = []v1.ContainerStatus{
						{
							Name: runningPod.Spec.Containers[0].Name,
							AllocatedResources: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse(allocCPU),
								v1.ResourceMemory: runningPod.Spec.Containers[0].Resources.Requests[v1.ResourceMemory],
							},
							Resources: &v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse(allocCPU),
									v1.ResourceMemory: runningPod.Spec.Containers[0].Resources.Requests[v1.ResourceMemory],
								},
							},
						},
					}
					runningPod, err = cs.CoreV1().Pods(testCtx.NS.Name).UpdateStatus(testCtx.Ctx, runningPod, metav1.UpdateOptions{})
					if err != nil {
						t.Fatalf("Failed updating allocated resources for victim %v: %v", podCopy.Name, err)
					}
				}
				if tt.terminatingByPreemptionPods[podCopy.Name] {
					runningPod.Status.Conditions = append(runningPod.Status.Conditions, v1.PodCondition{
						Type:   v1.DisruptionTarget,
						Status: v1.ConditionTrue,
						Reason: v1.PodReasonPreemptionByScheduler,
					})
					runningPod, err = cs.CoreV1().Pods(testCtx.NS.Name).UpdateStatus(testCtx.Ctx, runningPod, metav1.UpdateOptions{})
					if err != nil {
						t.Fatalf("Failed setting DisruptionTarget on %v: %v", podCopy.Name, err)
					}
					if err := cs.CoreV1().Pods(testCtx.NS.Name).Delete(testCtx.Ctx, runningPod.Name, metav1.DeleteOptions{GracePeriodSeconds: ptr.To[int64](30)}); err != nil {
						t.Fatalf("Failed initiating graceful deletion of %v: %v", podCopy.Name, err)
					}
				}
				pods = append(pods, runningPod)
			}

			// Create preemptor/resizing pod already scheduled to nodeName
			preemptorCfg := *tt.preemptorConfig
			preemptorCfg.NodeName = nodeName
			preemptorCfg.Namespace = testCtx.NS.Name
			preemptorPod := initPausePod(&preemptorCfg)
			if len(preemptorCfg.Resources.Claims) > 0 {
				claimName := preemptorCfg.Resources.Claims[0].Name
				preemptorPod.Spec.ResourceClaims = []v1.PodResourceClaim{
					{
						Name:              claimName,
						ResourceClaimName: &claimName,
					},
				}
			}
			preemptorPod, err := cs.CoreV1().Pods(testCtx.NS.Name).Create(testCtx.Ctx, preemptorPod, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("Failed to create preemptor pod: %v", err)
			}
			pods = append(pods, preemptorPod)

			scheduleAttemptsBefore := getScheduleAttemptsTotal(t)
			unschedulablePodsBefore, _ := testutil.GetGaugeMetricValue(schedmetrics.UnschedulablePods())
			preemptionAttemptsBefore, _ := testutil.GetCounterMetricValue(schedmetrics.PreemptionAttempts)
			resizePreemptionAttemptsBefore, _ := testutil.GetCounterMetricValue(schedmetrics.ResizePreemptionAttempts)

			// Update status of preemptor pod to simulate it running and being deferred.
			// Allocated CPU/Mem is set to 100m/100, while Spec request is higher.
			preemptorPod.Status.Phase = v1.PodRunning
			preemptorPod.Status.Conditions = []v1.PodCondition{
				{
					Type:   v1.PodScheduled,
					Status: v1.ConditionTrue,
				},
				{
					Type:   v1.PodResizePending,
					Status: v1.ConditionTrue,
					Reason: v1.PodReasonDeferred,
				},
			}
			preemptorPod.Status.ContainerStatuses = []v1.ContainerStatus{
				{
					Name: preemptorPod.Name,
					AllocatedResources: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(100, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(100, resource.DecimalSI),
					},
					Resources: &v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    *resource.NewMilliQuantity(100, resource.DecimalSI),
							v1.ResourceMemory: *resource.NewQuantity(100, resource.DecimalSI),
						},
					},
				},
			}
			preemptorPod.Status.NodeAllocatableResourceClaimStatuses = tt.preemptorNodeAllocatableClaimStatuses
			preemptorPod, err = cs.CoreV1().Pods(testCtx.NS.Name).UpdateStatus(testCtx.Ctx, preemptorPod, metav1.UpdateOptions{})
			if err != nil {
				t.Fatalf("Failed to update status of preemptor pod: %v", err)
			}

			queue := testCtx.Scheduler.SchedulingQueue

			// Wait for expected evictions (if any) or timeout
			if len(tt.expectEvictedNames) > 0 {
				evictedSet := make(map[string]bool, len(tt.expectEvictedNames))
				for _, name := range tt.expectEvictedNames {
					evictedSet[name] = true
					err = wait.PollUntilContextTimeout(testCtx.Ctx, 50*time.Millisecond, 10*time.Second, false,
						podIsGettingEvicted(cs, testCtx.NS.Name, name))
					if err != nil {
						t.Errorf("Expected pod %q to be evicted/deleted, but it was not", name)
					}
				}
				// Verify non-evicted pods (e.g., reprieved victims) were NOT evicted
				for _, p := range tt.existingPods {
					if evictedSet[p.Name] {
						continue
					}
					livePod, err := cs.CoreV1().Pods(testCtx.NS.Name).Get(testCtx.Ctx, p.Name, metav1.GetOptions{})
					if err == nil && livePod.DeletionTimestamp != nil {
						t.Errorf("Expected pod %q NOT to be evicted, but DeletionTimestamp is set", p.Name)
					}
				}

				// While victims are still terminating (GracePeriodSeconds=30), preemptor should remain in queue
				// without SchedulerEvaluationCompleted.
				qPod, found := queue.GetPod(testCtx.Ctx, preemptorPod.Name, preemptorPod.Namespace, nil)
				if !found {
					t.Fatalf("Expected preemptor pod to remain in scheduling queue while victims are terminating")
				}

				// Optional: trigger a second scheduling attempt while the evicted victim is still terminating
				// and verify PodEligibleToPreemptOthers suppresses double preemption of remaining victims.
				if tt.resizePodWhileVictimTerminating != "" {
					initialAttempts := qPod.Attempts
					targetPod, err := cs.CoreV1().Pods(testCtx.NS.Name).Get(testCtx.Ctx, tt.resizePodWhileVictimTerminating, metav1.GetOptions{})
					if err != nil {
						t.Fatalf("Failed to get %q: %v", tt.resizePodWhileVictimTerminating, err)
					}
					targetPod.Spec.Containers[0].Resources.Requests[v1.ResourceCPU] = resource.MustParse(tt.resizePodNewCPU)
					if _, err := cs.CoreV1().Pods(testCtx.NS.Name).UpdateResize(testCtx.Ctx, targetPod.Name, targetPod, metav1.UpdateOptions{}); err != nil {
						t.Fatalf("Failed to resize %q: %v", targetPod.Name, err)
					}
					err = wait.PollUntilContextTimeout(testCtx.Ctx, 50*time.Millisecond, 5*time.Second, false, func(ctx context.Context) (bool, error) {
						qPod, found := queue.GetPod(ctx, preemptorPod.Name, preemptorPod.Namespace, nil)
						return found && qPod.Attempts > initialAttempts, nil
					})
					if err != nil {
						t.Fatalf("Expected preemptor pod to run a second scheduling attempt while victim is terminating: %v", err)
					}
					for _, p := range tt.existingPods {
						if evictedSet[p.Name] {
							continue
						}
						livePod, err := cs.CoreV1().Pods(testCtx.NS.Name).Get(testCtx.Ctx, p.Name, metav1.GetOptions{})
						if err == nil && livePod.DeletionTimestamp != nil {
							t.Errorf("Expected remaining pod %q NOT to be preempted while first victim is terminating", p.Name)
						}
					}
				}

				// Complete victim deletion (simulating Kubelet finishing victim termination) and verify
				// the preemptor wakes up, sets SchedulerEvaluationCompleted=True, and leaves the queue.
				for _, name := range tt.expectEvictedNames {
					if err := cs.CoreV1().Pods(testCtx.NS.Name).Delete(testCtx.Ctx, name, metav1.DeleteOptions{GracePeriodSeconds: ptr.To[int64](0)}); err != nil {
						t.Fatalf("Failed to force-delete evicted victim %q: %v", name, err)
					}
				}
				err = wait.PollUntilContextTimeout(testCtx.Ctx, 50*time.Millisecond, 10*time.Second, false, func(ctx context.Context) (bool, error) {
					livePod, err := cs.CoreV1().Pods(testCtx.NS.Name).Get(ctx, preemptorPod.Name, metav1.GetOptions{})
					if err != nil {
						return false, err
					}
					_, inQueue := queue.GetPod(ctx, preemptorPod.Name, preemptorPod.Namespace, nil)
					return compresource.IsPodResizeSchedulerEvaluationCompleted(livePod) && !inQueue, nil
				})
				if err != nil {
					t.Errorf("Expected preemptor pod to have SchedulerEvaluationCompleted=True and leave queue after victim deletion: %v", err)
				}
			} else if tt.expectEvaluationCompletedImmediately {
				err = wait.PollUntilContextTimeout(testCtx.Ctx, 50*time.Millisecond, 10*time.Second, false, func(ctx context.Context) (bool, error) {
					livePod, err := cs.CoreV1().Pods(testCtx.NS.Name).Get(ctx, preemptorPod.Name, metav1.GetOptions{})
					if err != nil {
						return false, err
					}
					_, inQueue := queue.GetPod(ctx, preemptorPod.Name, preemptorPod.Namespace, nil)
					return compresource.IsPodResizeSchedulerEvaluationCompleted(livePod) && !inQueue, nil
				})
				if err != nil {
					t.Errorf("Expected preemptor pod to immediately set SchedulerEvaluationCompleted=True and leave queue: %v", err)
				}
				for _, p := range tt.existingPods {
					livePod, err := cs.CoreV1().Pods(testCtx.NS.Name).Get(testCtx.Ctx, p.Name, metav1.GetOptions{})
					if err == nil && livePod.DeletionTimestamp != nil {
						t.Errorf("Expected pod %q NOT to be evicted/deleted, but its DeletionTimestamp is set", p.Name)
					}
				}
			} else {
				// Wait for the scheduling attempt to complete and park the pod in the unschedulable queue,
				// then confirm non-terminating victims were not evicted and scheduler_pending_resize_pods reflects it.
				err = wait.PollUntilContextTimeout(testCtx.Ctx, 50*time.Millisecond, 5*time.Second, false, func(ctx context.Context) (bool, error) {
					qPod, found := queue.GetPod(ctx, preemptorPod.Name, preemptorPod.Namespace, nil)
					unschedResizePods, _ := testutil.GetGaugeMetricValue(schedmetrics.UnschedulableResizePods())
					return found && qPod.Attempts >= 1 && unschedResizePods >= 1, nil
				})
				if err != nil {
					t.Errorf("Expected unschedulable preemptor pod to complete at least 1 scheduling attempt and increment scheduler_pending_resize_pods{queue=\"unschedulable\"}: %v", err)
				}
				if unschedulablePods, err := testutil.GetGaugeMetricValue(schedmetrics.UnschedulablePods()); err != nil || unschedulablePods != unschedulablePodsBefore {
					t.Errorf("Expected scheduler_pending_pods{queue=\"unschedulable\"} to stay at %v, got %v (err: %v)", unschedulablePodsBefore, unschedulablePods, err)
				}
				for _, p := range tt.existingPods {
					if tt.terminatingByPreemptionPods[p.Name] {
						continue
					}
					livePod, err := cs.CoreV1().Pods(testCtx.NS.Name).Get(testCtx.Ctx, p.Name, metav1.GetOptions{})
					if err == nil && livePod.DeletionTimestamp != nil {
						t.Errorf("Expected pod %q NOT to be evicted/deleted, but its DeletionTimestamp is set", p.Name)
					}
				}
			}

			if tt.expectFailedSchedulingEvent {
				err = wait.PollUntilContextTimeout(testCtx.Ctx, 50*time.Millisecond, 5*time.Second, false, func(ctx context.Context) (bool, error) {
					evList, err := cs.EventsV1().Events(testCtx.NS.Name).List(ctx, metav1.ListOptions{})
					if err != nil {
						return false, err
					}
					for _, ev := range evList.Items {
						if ev.Regarding.UID == preemptorPod.UID && ev.Reason == "FailedScheduling" && ev.Type == v1.EventTypeWarning {
							return true, nil
						}
					}
					return false, nil
				})
				if err != nil {
					t.Errorf("Expected FailedScheduling warning event for preemptor pod %q, but none was found", preemptorPod.Name)
				}
			}

			if got := getScheduleAttemptsTotal(t) - scheduleAttemptsBefore; got != 0 {
				t.Errorf("Expected scheduler_schedule_attempts_total delta to be 0 for deferred resize pod, got %v", got)
			}
			preemptionAttemptsAfter, _ := testutil.GetCounterMetricValue(schedmetrics.PreemptionAttempts)
			if got := preemptionAttemptsAfter - preemptionAttemptsBefore; got != 0 {
				t.Errorf("Expected PreemptionAttempts delta to be 0 for deferred resize pod, got %v", got)
			}
			resizePreemptionAttemptsAfter, _ := testutil.GetCounterMetricValue(schedmetrics.ResizePreemptionAttempts)
			if !tt.expectEvaluationCompletedImmediately {
				if got := resizePreemptionAttemptsAfter - resizePreemptionAttemptsBefore; got < 1 {
					t.Errorf("Expected ResizePreemptionAttempts delta >= 1 when deferred resize triggers PostFilter, got %v", got)
				}
			}

			// Verify NominatedNodeName on preemptor pod status (retains empty for deferred pods)
			updatedPreemptor, err := cs.CoreV1().Pods(testCtx.NS.Name).Get(testCtx.Ctx, preemptorPod.Name, metav1.GetOptions{})
			if err != nil {
				t.Fatalf("Failed to get updated preemptor pod: %v", err)
			}
			if updatedPreemptor.Status.NominatedNodeName != "" {
				t.Errorf("Expected NominatedNodeName to remain empty, but got %q", updatedPreemptor.Status.NominatedNodeName)
			}

			// Verify PodScheduled condition remains True (retained running state)
			var gotScheduledStatus v1.ConditionStatus
			for _, cond := range updatedPreemptor.Status.Conditions {
				if cond.Type == v1.PodScheduled {
					gotScheduledStatus = cond.Status
				}
			}
			if gotScheduledStatus != v1.ConditionTrue {
				t.Errorf("Expected PodScheduled condition status to remain True, got %v", gotScheduledStatus)
			}
		})
	}
}

// registerResizeMetrics registers the metrics that metrics.Register skips while
// InPlacePodVerticalScalingSchedulerPreemption is disabled. Register runs once per process,
// so an earlier test may have run it with the gate off.
func registerResizeMetrics() {
	schedmetrics.Register()
	// Each returns AlreadyRegisteredError if Register ran with the gate on.
	_ = legacyregistry.Register(schedmetrics.PendingResizePods)
	_ = legacyregistry.Register(schedmetrics.ResizeSchedulingAlgorithmLatency)
	_ = legacyregistry.Register(schedmetrics.ResizePreemptionAttempts)
	_ = legacyregistry.Register(schedmetrics.ResizePreemptionVictims)
}

func getScheduleAttemptsTotal(t *testing.T) float64 {
	t.Helper()
	mfs, err := schedmetrics.GetGather().Gather()
	if err != nil {
		t.Fatalf("Failed to gather metrics: %v", err)
	}
	var total float64
	for _, mf := range mfs {
		if mf.GetName() == "scheduler_schedule_attempts_total" {
			for _, m := range mf.GetMetric() {
				total += m.GetCounter().GetValue()
			}
		}
	}
	return total
}

func setUpPreemptionTestWithContext(t *testing.T, testCtx *testutils.TestContext, idx int, nodeCPUCapacity, otherPodCPURequest, deferredPodCPURequest, deferredPodCPUAllocated string, disableResizePreemption bool) (*v1.Pod, *v1.Pod, *v1.Pod) {
	t.Helper()
	cs := testCtx.ClientSet

	nodeName1 := fmt.Sprintf("hint-node1-%d", idx)
	nodeName2 := fmt.Sprintf("hint-node2-%d", idx)

	// Create node1
	nodeObject1 := st.MakeNode().Name(nodeName1).Capacity(map[v1.ResourceName]string{
		v1.ResourcePods: "32",
		v1.ResourceCPU:  nodeCPUCapacity,
	}).Obj()
	if disableResizePreemption {
		nodeObject1.Spec.PodPreemptionPolicy = &v1.NodePodPreemptionPolicy{
			DisableResizePreemption: []string{"test-policy"},
		}
	}
	if _, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, nodeObject1, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create node1: %v", err)
	}
	t.Cleanup(func() {
		if err := cs.CoreV1().Nodes().Delete(testCtx.Ctx, nodeName1, metav1.DeleteOptions{}); err != nil && !apierrors.IsNotFound(err) {
			t.Errorf("Failed to delete node %s: %v", nodeName1, err)
		}
	})

	// Create node2 (irrelevant node)
	nodeObject2 := st.MakeNode().Name(nodeName2).Capacity(map[v1.ResourceName]string{
		v1.ResourcePods: "32",
		v1.ResourceCPU:  "500m",
	}).Obj()
	if _, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, nodeObject2, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create node2: %v", err)
	}
	t.Cleanup(func() {
		if err := cs.CoreV1().Nodes().Delete(testCtx.Ctx, nodeName2, metav1.DeleteOptions{}); err != nil && !apierrors.IsNotFound(err) {
			t.Errorf("Failed to delete node %s: %v", nodeName2, err)
		}
	})

	// Create 'other-pod' utilizing node1
	other := initPausePod(&testutils.PausePodConfig{
		Name:     fmt.Sprintf("other-pod-%d", idx),
		NodeName: nodeName1,
		Priority: &asyncframework.HighPriority,
		Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
			v1.ResourceCPU: resource.MustParse(otherPodCPURequest)},
		},
	})
	other.Namespace = testCtx.NS.Name
	other, err := runPausePod(cs, other)
	if err != nil {
		t.Fatalf("Failed to run other pod: %v", err)
	}

	// Create 'irrelevant-pod' utilizing node2
	irrelevant := initPausePod(&testutils.PausePodConfig{
		Name:     fmt.Sprintf("irrelevant-pod-%d", idx),
		NodeName: nodeName2,
		Priority: &asyncframework.HighPriority,
		Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
			v1.ResourceCPU: *resource.NewMilliQuantity(200, resource.DecimalSI)},
		},
	})
	irrelevant.Namespace = testCtx.NS.Name
	irrelevant, err = runPausePod(cs, irrelevant)
	if err != nil {
		t.Fatalf("Failed to run irrelevant pod: %v", err)
	}

	// Create deferred pod
	pod := initPausePod(&testutils.PausePodConfig{
		Name:     fmt.Sprintf("deferred-pod-%d", idx),
		NodeName: nodeName1,
		Priority: &asyncframework.LowPriority,
		Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
			v1.ResourceCPU: resource.MustParse(deferredPodCPURequest)},
		},
	})
	pod.Namespace = testCtx.NS.Name
	pod, err = cs.CoreV1().Pods(testCtx.NS.Name).Create(testCtx.Ctx, pod, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create pod: %v", err)
	}
	t.Cleanup(func() {
		testutils.CleanupPods(testCtx.Ctx, cs, t, []*v1.Pod{pod, other, irrelevant})
	})

	pod.Status.Phase = v1.PodRunning
	pod.Status.Conditions = []v1.PodCondition{
		{
			Type:   v1.PodScheduled,
			Status: v1.ConditionTrue,
		},
		{
			Type:   v1.PodResizePending,
			Status: v1.ConditionTrue,
			Reason: v1.PodReasonDeferred,
		},
	}
	pod.Status.ContainerStatuses = []v1.ContainerStatus{
		{
			Name: pod.Name,
			AllocatedResources: v1.ResourceList{
				v1.ResourceCPU: resource.MustParse(deferredPodCPUAllocated),
			},
			Resources: &v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceCPU: resource.MustParse(deferredPodCPUAllocated),
				},
			},
		},
	}
	pod, err = cs.CoreV1().Pods(testCtx.NS.Name).UpdateStatus(testCtx.Ctx, pod, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("Failed to update status: %v", err)
	}

	// Wait until deferred-pod fails scheduling and is parked in Unschedulable queue.
	queue := testCtx.Scheduler.SchedulingQueue
	err = wait.PollUntilContextTimeout(testCtx.Ctx, 50*time.Millisecond, 5*time.Second, false, func(context.Context) (bool, error) {
		unsched := queue.UnschedulablePods()
		for _, p := range unsched {
			if p.Name == pod.Name {
				return true, nil
			}
		}
		return false, nil
	})
	if err != nil {
		t.Fatalf("Expected deferred pod to be parked in Unschedulable queue, but it was not found")
	}

	return pod, other, irrelevant
}

func TestDeferredResizeQueueingHints(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.InPlacePodVerticalScaling:                    true,
		features.InPlacePodVerticalScalingSchedulerPreemption: true,
	})

	cfg := configtesting.V1ToInternalWithDefaults(t, configv1.KubeSchedulerConfiguration{
		Profiles: []configv1.KubeSchedulerProfile{{
			SchedulerName: new(v1.DefaultSchedulerName),
		}},
	})

	testCtx := testutils.InitTestSchedulerWithOptions(t,
		testutils.InitTestAPIServer(t, "def-q-hint", nil),
		0,
		scheduler.WithProfiles(cfg.Profiles...),
	)
	defer testCtx.SchedulerCloseFn()
	testutils.SyncSchedulerInformerFactory(testCtx)
	go testCtx.Scheduler.Run(testCtx.Ctx)

	tests := []struct {
		name               string
		nodeCPUCapacity    string
		otherPodCPURequest string
		deferredPodRequest string
		expectIncrement    bool
		trigger            func(ctx context.Context, cs clientset.Interface, ns string, pod, other, irrelevant *v1.Pod) error
	}{
		{
			name:            "irrelevant pod scale down ignores queue",
			expectIncrement: false,
			trigger: func(ctx context.Context, cs clientset.Interface, ns string, pod, other, irrelevant *v1.Pod) error {
				p, err := cs.CoreV1().Pods(ns).Get(ctx, irrelevant.Name, metav1.GetOptions{})
				if err != nil {
					return err
				}
				p.Spec.Containers[0].Resources.Requests = v1.ResourceList{v1.ResourceCPU: *resource.NewMilliQuantity(100, resource.DecimalSI)}
				_, err = cs.CoreV1().Pods(ns).UpdateResize(ctx, p.Name, p, metav1.UpdateOptions{})
				return err
			},
		},
		{
			name:            "assigned pod scale down on same node wakes queue",
			expectIncrement: true,
			trigger: func(ctx context.Context, cs clientset.Interface, ns string, pod, other, irrelevant *v1.Pod) error {
				p, err := cs.CoreV1().Pods(ns).Get(ctx, other.Name, metav1.GetOptions{})
				if err != nil {
					return err
				}
				p.Spec.Containers[0].Resources.Requests = v1.ResourceList{v1.ResourceCPU: *resource.NewMilliQuantity(100, resource.DecimalSI)}
				_, err = cs.CoreV1().Pods(ns).UpdateResize(ctx, p.Name, p, metav1.UpdateOptions{})
				return err
			},
		},
		{
			name:            "irrelevant pod deletion ignores queue",
			expectIncrement: false,
			trigger: func(ctx context.Context, cs clientset.Interface, ns string, pod, other, irrelevant *v1.Pod) error {
				return cs.CoreV1().Pods(ns).Delete(ctx, irrelevant.Name, metav1.DeleteOptions{GracePeriodSeconds: ptr.To[int64](0)})
			},
		},
		{
			name:            "assigned pod deletion on same node wakes queue",
			expectIncrement: true,
			trigger: func(ctx context.Context, cs clientset.Interface, ns string, pod, other, irrelevant *v1.Pod) error {
				return cs.CoreV1().Pods(ns).Delete(ctx, other.Name, metav1.DeleteOptions{GracePeriodSeconds: ptr.To[int64](0)})
			},
		},
		{
			name:            "non-resource label change ignores queue",
			expectIncrement: false,
			trigger: func(ctx context.Context, cs clientset.Interface, ns string, pod, other, irrelevant *v1.Pod) error {
				p, err := cs.CoreV1().Pods(ns).Get(ctx, other.Name, metav1.GetOptions{})
				if err != nil {
					return err
				}
				if p.Labels == nil {
					p.Labels = make(map[string]string)
				}
				p.Labels["updated-by-test"] = "true"
				_, err = cs.CoreV1().Pods(ns).Update(ctx, p, metav1.UpdateOptions{})
				return err
			},
		},
		{
			name:            "assigned node capacity increase wakes queue",
			expectIncrement: true,
			trigger: func(ctx context.Context, cs clientset.Interface, ns string, pod, other, irrelevant *v1.Pod) error {
				nodeName := pod.Spec.NodeName
				n, err := cs.CoreV1().Nodes().Get(ctx, nodeName, metav1.GetOptions{})
				if err != nil {
					return err
				}
				n.Status.Capacity[v1.ResourceCPU] = *resource.NewMilliQuantity(1000, resource.DecimalSI)
				n.Status.Allocatable[v1.ResourceCPU] = *resource.NewMilliQuantity(1000, resource.DecimalSI)
				_, err = cs.CoreV1().Nodes().UpdateStatus(ctx, n, metav1.UpdateOptions{})
				return err
			},
		},
		{
			name:            "target pod spec scale down wakes queue",
			expectIncrement: true,
			trigger: func(ctx context.Context, cs clientset.Interface, ns string, pod, other, irrelevant *v1.Pod) error {
				p, err := cs.CoreV1().Pods(ns).Get(ctx, pod.Name, metav1.GetOptions{})
				if err != nil {
					return err
				}
				p.Spec.Containers[0].Resources.Requests = v1.ResourceList{v1.ResourceCPU: *resource.NewMilliQuantity(80, resource.DecimalSI)}
				_, err = cs.CoreV1().Pods(ns).UpdateResize(ctx, p.Name, p, metav1.UpdateOptions{})
				return err
			},
		},
	}

	for idx, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			nodeCap := "200m"
			if tc.nodeCPUCapacity != "" {
				nodeCap = tc.nodeCPUCapacity
			}
			otherReq := "200m"
			if tc.otherPodCPURequest != "" {
				otherReq = tc.otherPodCPURequest
			}
			defReq := "100m"
			if tc.deferredPodRequest != "" {
				defReq = tc.deferredPodRequest
			}
			pod, other, irrelevant := setUpPreemptionTestWithContext(t, testCtx, idx, nodeCap, otherReq, defReq, "50m", false)

			cs := testCtx.ClientSet
			queue := testCtx.Scheduler.SchedulingQueue

			queuedPod, found := queue.GetPod(testCtx.Ctx, pod.Name, pod.Namespace, nil)
			if !found {
				t.Fatalf("Pod not found in queue")
			}
			initialAttempts := queuedPod.Attempts

			if err := tc.trigger(testCtx.Ctx, cs, testCtx.NS.Name, pod, other, irrelevant); err != nil {
				t.Fatalf("Failed triggering event: %v", err)
			}

			if tc.expectIncrement {
				// If the trigger made the pod fit (e.g. node capacity increase or other pod deletion),
				// the pod will complete Bind (SchedulerEvaluationCompleted=True) and leave the queue.
				err := wait.PollUntilContextTimeout(testCtx.Ctx, 50*time.Millisecond, 5*time.Second, false, func(ctx context.Context) (bool, error) {
					qPod, found := queue.GetPod(ctx, pod.Name, pod.Namespace, nil)
					if found && qPod.Attempts > initialAttempts {
						return true, nil
					}
					if !found {
						livePod, err := cs.CoreV1().Pods(testCtx.NS.Name).Get(ctx, pod.Name, metav1.GetOptions{})
						if err == nil && compresource.IsPodResizeSchedulerEvaluationCompleted(livePod) {
							return true, nil
						}
					}
					return false, nil
				})
				if err != nil {
					t.Fatalf("Expected queue attempts to increment or pod evaluation to complete after trigger")
				}
			} else {
				time.Sleep(300 * time.Millisecond)
				qPod, found := queue.GetPod(testCtx.Ctx, pod.Name, pod.Namespace, nil)
				if found && qPod.Attempts > initialAttempts {
					t.Fatalf("Expected attempts not to increment, but went from %d to %d", initialAttempts, qPod.Attempts)
				}
			}
		})
	}
}

func TestDeferredResizeNodePreemptionPolicy(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.InPlacePodVerticalScaling:                    true,
		features.InPlacePodVerticalScalingSchedulerPreemption: true,
	})

	cfg := configtesting.V1ToInternalWithDefaults(t, configv1.KubeSchedulerConfiguration{
		Profiles: []configv1.KubeSchedulerProfile{{
			SchedulerName: new(v1.DefaultSchedulerName),
		}},
	})

	testCtx := testutils.InitTestSchedulerWithOptions(t,
		testutils.InitTestAPIServer(t, "def-policy", nil),
		0,
		scheduler.WithProfiles(cfg.Profiles...),
	)
	defer testCtx.SchedulerCloseFn()
	testutils.SyncSchedulerInformerFactory(testCtx)
	go testCtx.Scheduler.Run(testCtx.Ctx)

	tests := []struct {
		name                             string
		initiallyDisableResizePreemption bool
		targetIrrelevantNode             bool
		newPolicy                        *v1.NodePodPreemptionPolicy
		expectIncrement                  bool
	}{
		{
			name:                             "disabling node preemption policy does not wake queue",
			initiallyDisableResizePreemption: false,
			newPolicy: &v1.NodePodPreemptionPolicy{
				DisableResizePreemption: []string{"test-policy"},
			},
			expectIncrement: false,
		},
		{
			name:                             "enabling node preemption policy wakes queue",
			initiallyDisableResizePreemption: true,
			newPolicy:                        nil,
			expectIncrement:                  true,
		},
		{
			name:                             "irrelevant node preemption policy change ignores queue",
			initiallyDisableResizePreemption: false,
			targetIrrelevantNode:             true,
			newPolicy: &v1.NodePodPreemptionPolicy{
				DisableResizePreemption: []string{"test-policy"},
			},
			expectIncrement: false,
		},
	}

	for idx, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			pod, _, irrelevant := setUpPreemptionTestWithContext(t, testCtx, idx, "200m", "200m", "100m", "50m", tc.initiallyDisableResizePreemption)

			cs := testCtx.ClientSet
			queue := testCtx.Scheduler.SchedulingQueue

			queuedPod, found := queue.GetPod(testCtx.Ctx, pod.Name, pod.Namespace, nil)
			if !found {
				t.Fatalf("Expected pod to be in queue initially")
			}
			initialAttempts := queuedPod.Attempts

			targetNodeName := pod.Spec.NodeName
			if tc.targetIrrelevantNode {
				targetNodeName = irrelevant.Spec.NodeName
			}
			n, err := cs.CoreV1().Nodes().Get(testCtx.Ctx, targetNodeName, metav1.GetOptions{})
			if err != nil {
				t.Fatalf("Failed to get node %q: %v", targetNodeName, err)
			}
			n.Spec.PodPreemptionPolicy = tc.newPolicy
			if _, err := cs.CoreV1().Nodes().Update(testCtx.Ctx, n, metav1.UpdateOptions{}); err != nil {
				t.Fatalf("Failed to update node %q: %v", targetNodeName, err)
			}

			if tc.expectIncrement {
				err = wait.PollUntilContextTimeout(testCtx.Ctx, 50*time.Millisecond, 5*time.Second, false, func(ctx context.Context) (bool, error) {
					qPod, found := queue.GetPod(ctx, pod.Name, pod.Namespace, nil)
					return found && qPod.Attempts > initialAttempts, nil
				})
				if err != nil {
					t.Fatalf("Expected attempts to increment after updating node preemption policy")
				}
			} else {
				time.Sleep(300 * time.Millisecond)
				qPod, found := queue.GetPod(testCtx.Ctx, pod.Name, pod.Namespace, nil)
				if !found || qPod.Attempts > initialAttempts {
					t.Fatalf("Expected attempts not to increment and pod to remain in queue, initial=%d, found=%v", initialAttempts, found)
				}
			}
		})
	}
}

func TestDeferredResizeAddPodHandlerOnStartup(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.InPlacePodVerticalScaling:                    true,
		features.InPlacePodVerticalScalingSchedulerPreemption: true,
	})

	cfg := configtesting.V1ToInternalWithDefaults(t, configv1.KubeSchedulerConfiguration{
		Profiles: []configv1.KubeSchedulerProfile{{
			SchedulerName: new(v1.DefaultSchedulerName),
		}},
	})

	sharedAPICtx := testutils.InitTestAPIServer(t, "def-startup", nil)
	nodeObject := st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{
		v1.ResourcePods: "32",
		v1.ResourceCPU:  "200m",
	}).Obj()
	if _, err := sharedAPICtx.ClientSet.CoreV1().Nodes().Create(sharedAPICtx.Ctx, nodeObject, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create node: %v", err)
	}

	tests := []struct {
		name          string
		conditions    []v1.PodCondition
		expectInQueue bool
	}{
		{
			name: "AddPod handler enqueues existing deferred pod without SchedulerEvaluationCompleted on scheduler startup",
			conditions: []v1.PodCondition{
				{Type: v1.PodScheduled, Status: v1.ConditionTrue},
				{Type: v1.PodResizePending, Status: v1.ConditionTrue, Reason: v1.PodReasonDeferred},
			},
			expectInQueue: true,
		},
		{
			name: "AddPod handler skips existing deferred pod when SchedulerEvaluationCompleted=True on scheduler startup",
			conditions: []v1.PodCondition{
				{Type: v1.PodScheduled, Status: v1.ConditionTrue},
				{Type: v1.PodResizePending, Status: v1.ConditionTrue, Reason: v1.PodReasonDeferred},
				{Type: v1.SchedulerEvaluationCompleted, Status: v1.ConditionTrue, Reason: v1.PodReasonDeferredPodResize},
			},
			expectInQueue: false,
		},
	}

	for idx, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			childCtx := withNewNamespace(t, sharedAPICtx, "def-startup")
			cs := childCtx.ClientSet

			pod := initPausePod(&testutils.PausePodConfig{
				Name:      fmt.Sprintf("deferred-pod-%d", idx),
				Namespace: childCtx.NS.Name,
				NodeName:  "node1",
				Priority:  &asyncframework.LowPriority,
				Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
					v1.ResourceCPU: *resource.NewMilliQuantity(100, resource.DecimalSI)},
				},
			})
			pod, err := cs.CoreV1().Pods(childCtx.NS.Name).Create(childCtx.Ctx, pod, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("Failed to create pod: %v", err)
			}
			t.Cleanup(func() {
				testutils.CleanupPods(childCtx.Ctx, cs, t, []*v1.Pod{pod})
			})

			pod.Status.Phase = v1.PodRunning
			pod.Status.Conditions = tc.conditions
			pod.Status.ContainerStatuses = []v1.ContainerStatus{
				{
					Name:               pod.Name,
					AllocatedResources: v1.ResourceList{v1.ResourceCPU: *resource.NewMilliQuantity(50, resource.DecimalSI)},
					Resources:          &v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceCPU: *resource.NewMilliQuantity(50, resource.DecimalSI)}},
				},
			}
			pod, err = cs.CoreV1().Pods(childCtx.NS.Name).UpdateStatus(childCtx.Ctx, pod, metav1.UpdateOptions{})
			if err != nil {
				t.Fatalf("Failed to update status: %v", err)
			}

			testCtx := testutils.InitTestSchedulerWithOptions(t, childCtx, 0, scheduler.WithProfiles(cfg.Profiles...))
			defer testCtx.SchedulerCloseFn()
			testutils.SyncSchedulerInformerFactory(testCtx)

			queue := testCtx.Scheduler.SchedulingQueue
			cache := testCtx.Scheduler.Cache
			if tc.expectInQueue {
				err = wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 2*time.Second, true, func(ctx context.Context) (bool, error) {
					_, found := queue.GetPod(ctx, pod.Name, pod.Namespace, nil)
					return found, nil
				})
				if err != nil {
					t.Fatalf("Expected pod to be queued automatically by AddPod handler on scheduler startup: %v", err)
				}
			} else {
				err = wait.PollUntilContextTimeout(testCtx.Ctx, 50*time.Millisecond, 2*time.Second, true, func(ctx context.Context) (bool, error) {
					_, errCache := cache.GetPod(pod)
					return errCache == nil, nil
				})
				if err != nil {
					t.Fatalf("Expected pod to be added to scheduler cache on startup: %v", err)
				}
				if _, found := queue.GetPod(testCtx.Ctx, pod.Name, pod.Namespace, nil); found {
					t.Fatalf("Expected pod with SchedulerEvaluationCompleted=True NOT to be queued on startup")
				}
			}
		})
	}
}

func TestDeferredResizeQueueingHandlers(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.InPlacePodVerticalScaling:                    true,
		features.InPlacePodVerticalScalingSchedulerPreemption: true,
	})

	cfg := configtesting.V1ToInternalWithDefaults(t, configv1.KubeSchedulerConfiguration{
		Profiles: []configv1.KubeSchedulerProfile{{
			SchedulerName: new(v1.DefaultSchedulerName),
		}},
	})

	testCtx := testutils.InitTestSchedulerWithOptions(t,
		testutils.InitTestAPIServer(t, "def-handlers", nil),
		0,
		scheduler.WithProfiles(cfg.Profiles...),
	)
	defer testCtx.SchedulerCloseFn()
	testutils.SyncSchedulerInformerFactory(testCtx)

	cs := testCtx.ClientSet
	queue := testCtx.Scheduler.SchedulingQueue
	cache := testCtx.Scheduler.Cache

	nodeObject := st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{
		v1.ResourcePods: "32",
		v1.ResourceCPU:  "200m",
	}).Obj()
	if _, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, nodeObject, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create node: %v", err)
	}

	tests := []struct {
		name                     string
		initialConditions        []v1.PodCondition
		expectInitiallyInQueue   bool
		updatedConditions        []v1.PodCondition
		deletePod                bool
		expectInQueueAfterAction bool
		expectRemovedFromCache   bool
	}{
		{
			name: "UpdatePod handler enqueues pod when it transitions to deferred resize",
			initialConditions: []v1.PodCondition{
				{Type: v1.PodScheduled, Status: v1.ConditionTrue},
			},
			expectInitiallyInQueue: false,
			updatedConditions: []v1.PodCondition{
				{Type: v1.PodScheduled, Status: v1.ConditionTrue},
				{Type: v1.PodResizePending, Status: v1.ConditionTrue, Reason: v1.PodReasonDeferred},
			},
			expectInQueueAfterAction: true,
		},
		{
			name: "UpdatePod handler removes pod from queue when deferred condition is cleared",
			initialConditions: []v1.PodCondition{
				{Type: v1.PodScheduled, Status: v1.ConditionTrue},
				{Type: v1.PodResizePending, Status: v1.ConditionTrue, Reason: v1.PodReasonDeferred},
			},
			expectInitiallyInQueue: true,
			updatedConditions: []v1.PodCondition{
				{Type: v1.PodScheduled, Status: v1.ConditionTrue},
			},
			expectInQueueAfterAction: false,
		},
		{
			name: "UpdatePod handler re-enqueues pod when Kubelet clears SchedulerEvaluationCompleted while Deferred remains",
			initialConditions: []v1.PodCondition{
				{Type: v1.PodScheduled, Status: v1.ConditionTrue},
				{Type: v1.PodResizePending, Status: v1.ConditionTrue, Reason: v1.PodReasonDeferred},
				{Type: v1.SchedulerEvaluationCompleted, Status: v1.ConditionTrue, Reason: v1.PodReasonDeferredPodResize},
			},
			expectInitiallyInQueue: false,
			updatedConditions: []v1.PodCondition{
				{Type: v1.PodScheduled, Status: v1.ConditionTrue},
				{Type: v1.PodResizePending, Status: v1.ConditionTrue, Reason: v1.PodReasonDeferred},
			},
			expectInQueueAfterAction: true,
		},
		{
			name: "DeletePod handler removes pod from both cache and scheduling queue when deleted",
			initialConditions: []v1.PodCondition{
				{Type: v1.PodScheduled, Status: v1.ConditionTrue},
				{Type: v1.PodResizePending, Status: v1.ConditionTrue, Reason: v1.PodReasonDeferred},
			},
			expectInitiallyInQueue:   true,
			deletePod:                true,
			expectInQueueAfterAction: false,
			expectRemovedFromCache:   true,
		},
	}

	for idx, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			pod := initPausePod(&testutils.PausePodConfig{
				Name:      fmt.Sprintf("deferred-pod-%d", idx),
				Namespace: testCtx.NS.Name,
				NodeName:  "node1",
				Priority:  &asyncframework.LowPriority,
				Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
					v1.ResourceCPU: *resource.NewMilliQuantity(100, resource.DecimalSI)},
				},
			})
			pod, err := cs.CoreV1().Pods(testCtx.NS.Name).Create(testCtx.Ctx, pod, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("Failed to create pod: %v", err)
			}
			t.Cleanup(func() {
				testutils.CleanupPods(testCtx.Ctx, cs, t, []*v1.Pod{pod})
			})

			pod.Status.Phase = v1.PodRunning
			pod.Status.Conditions = tc.initialConditions
			pod.Status.ContainerStatuses = []v1.ContainerStatus{
				{
					Name:               pod.Name,
					AllocatedResources: v1.ResourceList{v1.ResourceCPU: *resource.NewMilliQuantity(50, resource.DecimalSI)},
					Resources:          &v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceCPU: *resource.NewMilliQuantity(50, resource.DecimalSI)}},
				},
			}
			pod, err = cs.CoreV1().Pods(testCtx.NS.Name).UpdateStatus(testCtx.Ctx, pod, metav1.UpdateOptions{})
			if err != nil {
				t.Fatalf("Failed to update initial status: %v", err)
			}

			if tc.expectInitiallyInQueue {
				err = wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 2*time.Second, true, func(ctx context.Context) (bool, error) {
					_, found := queue.GetPod(ctx, pod.Name, pod.Namespace, nil)
					_, errCache := cache.GetPod(pod)
					return found && errCache == nil, nil
				})
				if err != nil {
					t.Fatalf("Expected pod to be in queue and cache initially: %v", err)
				}
			} else {
				err = wait.PollUntilContextTimeout(testCtx.Ctx, 50*time.Millisecond, 2*time.Second, true, func(ctx context.Context) (bool, error) {
					_, errCache := cache.GetPod(pod)
					return errCache == nil, nil
				})
				if err != nil {
					t.Fatalf("Expected pod to be added to scheduler cache initially: %v", err)
				}
				if _, found := queue.GetPod(testCtx.Ctx, pod.Name, pod.Namespace, nil); found {
					t.Fatalf("Expected pod NOT to be in queue initially")
				}
			}

			if tc.deletePod {
				if err := cs.CoreV1().Pods(testCtx.NS.Name).Delete(testCtx.Ctx, pod.Name, metav1.DeleteOptions{GracePeriodSeconds: ptr.To[int64](0)}); err != nil {
					t.Fatalf("Failed to delete pod: %v", err)
				}
			} else {
				pod, err = cs.CoreV1().Pods(testCtx.NS.Name).Get(testCtx.Ctx, pod.Name, metav1.GetOptions{})
				if err != nil {
					t.Fatalf("Failed to get pod: %v", err)
				}
				pod.Status.Conditions = tc.updatedConditions
				if _, err := cs.CoreV1().Pods(testCtx.NS.Name).UpdateStatus(testCtx.Ctx, pod, metav1.UpdateOptions{}); err != nil {
					t.Fatalf("Failed to update status: %v", err)
				}
			}

			err = wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 2*time.Second, true, func(ctx context.Context) (bool, error) {
				_, found := queue.GetPod(ctx, pod.Name, pod.Namespace, nil)
				if found != tc.expectInQueueAfterAction {
					return false, nil
				}
				if tc.expectRemovedFromCache {
					_, errCache := cache.GetPod(pod)
					return errCache != nil, nil
				}
				return true, nil
			})
			if err != nil {
				t.Fatalf("Expected pod in queue=%v (removedFromCache=%v) after action: %v", tc.expectInQueueAfterAction, tc.expectRemovedFromCache, err)
			}
		})
	}
}
