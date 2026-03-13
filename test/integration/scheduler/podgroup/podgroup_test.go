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

package podgroup

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	schedulingapi "k8s.io/api/scheduling/v1alpha2"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler"
	"k8s.io/kubernetes/pkg/scheduler/backend/podgroupmanager"
	"k8s.io/kubernetes/pkg/scheduler/backend/queue"
	"k8s.io/kubernetes/pkg/scheduler/framework/preemption"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	testutils "k8s.io/kubernetes/test/integration/util"
)

// podInUnschedulablePods checks if the given Pod is in the unschedulable pods pool.
func podInUnschedulablePods(t *testing.T, queue queue.SchedulingQueue, podName string) bool {
	t.Helper()
	unschedPods := queue.UnschedulablePods()
	for _, pod := range unschedPods {
		if pod.Name == podName {
			return true
		}
	}
	return false
}

func TestPodGroupScheduling(t *testing.T) {
	node := st.MakeNode().Name("node").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Obj()

	gangPodGroup := st.MakePodGroup().Name("pg1").TemplateRef("t1", "workload").MinCount(3).Obj()

	otherGangPodGroup := st.MakePodGroup().Name("pg2").TemplateRef("t", "other-workload").MinCount(3).Obj()

	basicPodGroup := st.MakePodGroup().Name("pg1").TemplateRef("t2", "workload").BasicPolicy().Obj()

	p1 := st.MakePod().Name("p1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").
		PodGroupName("pg1").Priority(100).Obj()
	p2 := st.MakePod().Name("p2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").
		PodGroupName("pg1").Priority(100).Obj()
	p3 := st.MakePod().Name("p3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").
		PodGroupName("pg1").Priority(100).Obj()
	p4 := st.MakePod().Name("p4").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").
		PodGroupName("pg1").Priority(100).Obj()

	blockerPod := st.MakePod().Name("blocker").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").
		ZeroTerminationGracePeriod().Priority(100).Obj()
	smallBlockerPod := st.MakePod().Name("small-blocker").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").
		ZeroTerminationGracePeriod().Priority(100).Obj()
	lowPriorityBlockerPod := st.MakePod().Name("low-priority-blocker").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").
		ZeroTerminationGracePeriod().Priority(10).Obj()

	lowP1 := st.MakePod().Name("low-p1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").
		ZeroTerminationGracePeriod().Priority(10).Obj()
	lowP2 := st.MakePod().Name("low-p2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").
		ZeroTerminationGracePeriod().Priority(10).Obj()
	lowP3 := st.MakePod().Name("low-p3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").
		ZeroTerminationGracePeriod().Priority(10).Obj()
	lowP4 := st.MakePod().Name("low-p4").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").
		ZeroTerminationGracePeriod().Priority(10).Obj()

	otherP1 := st.MakePod().Name("other-p1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").
		PodGroupName("pg2").Priority(100).Obj()
	otherP2 := st.MakePod().Name("other-p2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").
		PodGroupName("pg2").Priority(100).Obj()
	otherP3 := st.MakePod().Name("other-p3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").
		PodGroupName("pg2").Priority(100).Obj()

	type waitForAnyPodsScheduled struct {
		pods             []*v1.Pod
		numScheduled     int
		numUnschedulable int
	}

	// step represents a single step in a test scenario.
	type step struct {
		name                         string
		createPodGroup               *schedulingapi.PodGroup
		createPods                   []*v1.Pod
		deletePods                   []string
		waitForPodsGatedOnPreEnqueue []string
		waitForPodsUnschedulable     []string
		waitForPodsScheduled         []string
		waitForPodsRemoved           []string
		waitForAnyPodsScheduled      *waitForAnyPodsScheduled
	}

	tests := []struct {
		name                          string
		enableWorkloadAwarePreemption bool
		steps                         []step
	}{
		{
			name: "gang schedules when pod group and resources are available",
			steps: []step{
				{
					name:           "Create the PodGroup object",
					createPodGroup: gangPodGroup,
				},
				{
					name:       "Create all pods belonging to the gang",
					createPods: []*v1.Pod{p1, p2, p3},
				},
				{
					name:                 "Verify all gang pods are scheduled successfully",
					waitForPodsScheduled: []string{"p1", "p2", "p3"},
				},
			},
		},
		{
			name: "gang waits for quorum to start, then schedules",
			steps: []step{
				{
					name:           "Create the PodGroup object",
					createPodGroup: gangPodGroup,
				},
				{
					name:       "Create subset of pods belonging to the gang",
					createPods: []*v1.Pod{p1, p2},
				},
				{
					name:                         "Verify pods are gated at PreEnqueue (no quorum)",
					waitForPodsGatedOnPreEnqueue: []string{"p1", "p2"},
				},
				{
					name:       "Create the last pod belonging to the gang to unblock PreEnqueue",
					createPods: []*v1.Pod{p3},
				},
				{
					name:                 "Verify all gang pods are scheduled successfully",
					waitForPodsScheduled: []string{"p1", "p2", "p3"},
				},
			},
		},
		{
			name: "gang waits for pod group, then for resources, then schedules",
			steps: []step{
				{
					name:       "Create the resource-blocking pod",
					createPods: []*v1.Pod{blockerPod},
				},
				{
					name:                 "Schedule the resource-blocking pod",
					waitForPodsScheduled: []string{"blocker"},
				},
				{
					name:       "Create gang pods before PodGroup is created",
					createPods: []*v1.Pod{p1, p2, p3},
				},
				{
					name:                         "Verify pods are gated at PreEnqueue (no PodGroup object)",
					waitForPodsGatedOnPreEnqueue: []string{"p1", "p2", "p3"},
				},
				{
					name:           "Create the PodGroup to unblock PreEnqueue",
					createPodGroup: gangPodGroup,
				},
				{
					name:                     "Verify pods become unschedulable (Permit timeout due to resource blocker)",
					waitForPodsUnschedulable: []string{"p1", "p2", "p3"},
				},
				{
					name:       "Delete the resource-blocking pod",
					deletePods: []string{"blocker"},
				},
				{
					name:                 "Verify the entire gang is now scheduled",
					waitForPodsScheduled: []string{"p1", "p2", "p3"},
				},
			},
		},
		{
			name: "minCount is scheduled, but one pod from a gang remain unschedulable until the blocked resources are released",
			steps: []step{
				{
					name:       "Create the resource-blocking pod",
					createPods: []*v1.Pod{smallBlockerPod},
				},
				{
					name:                 "Schedule the resource-blocking pod",
					waitForPodsScheduled: []string{"small-blocker"},
				},
				{
					name:       "Create all pods belonging to the gang (more than minCount) before the PodGroup is created",
					createPods: []*v1.Pod{p1, p2, p3, p4},
				},
				{
					name:           "Create the PodGroup to unblock PreEnqueue",
					createPodGroup: gangPodGroup,
				},
				{
					name: "Verify minCount pods is scheduled successfully and one becomes unschedulable (resource-blocking pod is blocking the space)",
					waitForAnyPodsScheduled: &waitForAnyPodsScheduled{
						pods:             []*v1.Pod{p1, p2, p3, p4},
						numScheduled:     3,
						numUnschedulable: 1,
					},
				},
				{
					name:       "Delete the resource-blocking pod",
					deletePods: []string{"small-blocker"},
				},
				{
					name:                 "Verify the entire gang is now scheduled",
					waitForPodsScheduled: []string{"p1", "p2", "p3", "p4"},
				},
			},
		},
		{
			name: "two gangs competing for the same resources shouldn't deadlock, reversed order",
			steps: []step{
				{
					name:           "Create the PodGroup object",
					createPodGroup: gangPodGroup,
				},
				{
					name:           "Create the other PodGroup object",
					createPodGroup: otherGangPodGroup,
				},
				{
					name:       "Create pods from both gangs",
					createPods: []*v1.Pod{otherP3, p3, otherP2, p2, otherP1, p1},
				},
				{
					name:                 "Verify the entire other gang is now scheduled",
					waitForPodsScheduled: []string{"other-p1", "other-p2", "other-p3"},
				},
				{
					name:                     "Verify the entire gang becomes unschedulable",
					waitForPodsUnschedulable: []string{"p1", "p2", "p3"},
				},
			},
		},
		{
			name: "two gangs competing for the same resources shouldn't deadlock",
			steps: []step{
				{
					name:           "Create the PodGroup object",
					createPodGroup: gangPodGroup,
				},
				{
					name:           "Create the other PodGroup object",
					createPodGroup: otherGangPodGroup,
				},
				{
					name:       "Create pods from both gangs",
					createPods: []*v1.Pod{p1, otherP1, p2, otherP2, p3, otherP3},
				},
				{
					name:                 "Verify the entire gang is now scheduled",
					waitForPodsScheduled: []string{"p1", "p2", "p3"},
				},
				{
					name:                     "Verify the entire other gang becomes unschedulable",
					waitForPodsUnschedulable: []string{"other-p1", "other-p2", "other-p3"},
				},
			},
		},
		{
			name: "gang schedules with preemption",
			steps: []step{
				{
					name:       "Create a low priority pod taking all resources",
					createPods: []*v1.Pod{lowPriorityBlockerPod},
				},
				{
					name:                 "Schedule the low priority resource-blocking pod",
					waitForPodsScheduled: []string{"low-priority-blocker"},
				},
				{
					name:           "Create the PodGroup object",
					createPodGroup: gangPodGroup,
				},
				{
					name:       "Create high priority gang pods",
					createPods: []*v1.Pod{p1, p2, p3, p4},
				},
				{
					name:                 "Verify all gang pods are scheduled successfully (after preemption)",
					waitForPodsScheduled: []string{"p1", "p2", "p3", "p4"},
				},
				{
					name:               "Verify preemption victims were removed",
					waitForPodsRemoved: []string{"low-priority-blocker"},
				},
			},
		},
		{
			name: "basic group schedules when pod group and resources are available, without gang enforcement",
			steps: []step{
				{
					name:           "Create the PodGroup object",
					createPodGroup: basicPodGroup,
				},
				{
					name:       "Create one pod belonging to the group",
					createPods: []*v1.Pod{p1},
				},
				{
					name:                 "Verify group's pod is scheduled successfully",
					waitForPodsScheduled: []string{"p1"},
				},
				{
					name:       "Create another pods belonging to the group",
					createPods: []*v1.Pod{p2, p3},
				},
				{
					name:                 "Verify group's pods are scheduled successfully",
					waitForPodsScheduled: []string{"p1", "p2", "p3"},
				},
			},
		},
		{
			name: "basic group waits for pod group, part of it waits for resources, then schedules",
			steps: []step{
				{
					name:       "Create the resource-blocking pod",
					createPods: []*v1.Pod{blockerPod},
				},
				{
					name:                 "Schedule the resource-blocking pod",
					waitForPodsScheduled: []string{"blocker"},
				},
				{
					name:       "Create basic group pods before PodGroup is created",
					createPods: []*v1.Pod{p1, p2, p3},
				},
				{
					name:                         "Verify pods are gated at PreEnqueue (no PodGroup object)",
					waitForPodsGatedOnPreEnqueue: []string{"p1", "p2", "p3"},
				},
				{
					name:           "Create the PodGroup to unblock PreEnqueue",
					createPodGroup: basicPodGroup,
				},
				{
					name: "Verify two pods are scheduled successfully and one becomes unschedulable (resource-blocking pod is blocking the space)",
					waitForAnyPodsScheduled: &waitForAnyPodsScheduled{
						pods:             []*v1.Pod{p1, p2, p3},
						numScheduled:     2,
						numUnschedulable: 1,
					},
				},
				{
					name:       "Delete the resource-blocking pod",
					deletePods: []string{"blocker"},
				},
				{
					name:                 "Verify the entire group is now scheduled",
					waitForPodsScheduled: []string{"p1", "p2", "p3"},
				},
			},
		},
		{
			name: "basic group schedules with preemption",
			steps: []step{
				{
					name:       "Create a low priority pod taking all resources",
					createPods: []*v1.Pod{lowPriorityBlockerPod},
				},
				{
					name:                 "Schedule the low priority resource-blocking pod",
					waitForPodsScheduled: []string{"low-priority-blocker"},
				},
				{
					name:           "Create the PodGroup object",
					createPodGroup: basicPodGroup,
				},
				{
					name:       "Create high priority group's pods",
					createPods: []*v1.Pod{p1, p2, p3, p4},
				},
				{
					name:                 "Verify all group's pods are scheduled successfully (after preemption)",
					waitForPodsScheduled: []string{"p1", "p2", "p3", "p4"},
				},
				{
					name:               "Verify preemption victims were removed",
					waitForPodsRemoved: []string{"low-priority-blocker"},
				},
			},
		},
		{
			name:                          "gang schedules with workload-aware preemption",
			enableWorkloadAwarePreemption: true,
			steps: []step{
				{
					name:       "Create low priority pods that take up all node resources",
					createPods: []*v1.Pod{lowP1, lowP2, lowP3, lowP4},
				},
				{
					name:                 "Wait for all low priority pods to be scheduled",
					waitForPodsScheduled: []string{"low-p1", "low-p2", "low-p3", "low-p4"},
				},
				{
					name:           "Create the Workload object",
					createPodGroup: gangPodGroup,
				},
				{
					name:       "Create high priority gang pods",
					createPods: []*v1.Pod{p1, p2, p3, p4},
				},
				{
					name:                 "Verify all gang pods are scheduled successfully (after workload-aware preemption)",
					waitForPodsScheduled: []string{"p1", "p2", "p3", "p4"},
				},
				{
					name:               "Verify preemption victims were removed",
					waitForPodsRemoved: []string{"low-p1", "low-p2", "low-p3", "low-p4"},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload:         true,
				features.GangScheduling:          true,
				features.WorkloadAwarePreemption: tt.enableWorkloadAwarePreemption,
			})

			podgroupmanager.DefaultSchedulingTimeoutDuration = 5 * time.Second

			testCtx := testutils.InitTestSchedulerWithNS(t, "podgroup-scheduling",
				// disable backoff
				scheduler.WithPodMaxBackoffSeconds(0),
				scheduler.WithPodInitialBackoffSeconds(0))

			cs, ns := testCtx.ClientSet, testCtx.NS.Name

			_, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, node, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("Failed to create node: %v", err)
			}

			for i, step := range tt.steps {
				t.Logf("Executing step %d: %s", i, step.name)
				switch {
				case step.createPods != nil:
					for _, pod := range step.createPods {
						p := pod.DeepCopy()
						p.Namespace = ns
						if _, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, p, metav1.CreateOptions{}); err != nil {
							t.Fatalf("Step %d: Failed to create pod %s: %v", i, p.Name, err)
						}
					}
				case step.createPodGroup != nil:
					w := step.createPodGroup.DeepCopy()
					w.Namespace = ns
					if _, err := cs.SchedulingV1alpha2().PodGroups(ns).Create(testCtx.Ctx, w, metav1.CreateOptions{}); err != nil {
						t.Fatalf("Step %d: Failed to create pod group %s: %v", i, w.Name, err)
					}
					// Ensure all next steps will see this pod group.
					err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, wait.ForeverTestTimeout, false,
						func(_ context.Context) (bool, error) {
							_, err := testCtx.InformerFactory.Scheduling().V1alpha2().PodGroups().Lister().PodGroups(ns).Get(w.Name)
							if err != nil {
								if apierrors.IsNotFound(err) {
									return false, nil
								}
								return false, err
							}
							return true, nil
						},
					)
					if err != nil {
						t.Fatalf("Step %d: Failed to wait for pod group %s to be discoverable by scheduler: %v", i, w.Name, err)
					}
				case step.deletePods != nil:
					for _, podName := range step.deletePods {
						if err := cs.CoreV1().Pods(ns).Delete(testCtx.Ctx, podName, metav1.DeleteOptions{}); err != nil {
							t.Fatalf("Step %d: Failed to delete pod %s: %v", i, podName, err)
						}
					}
				case step.waitForPodsGatedOnPreEnqueue != nil:
					for _, podName := range step.waitForPodsGatedOnPreEnqueue {
						err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, wait.ForeverTestTimeout, false,
							func(_ context.Context) (bool, error) {
								return podInUnschedulablePods(t, testCtx.Scheduler.SchedulingQueue, podName), nil
							},
						)
						if err != nil {
							t.Fatalf("Step %d: Failed to wait for pod %s to be in unschedulable pods pool: %v", i, podName, err)
						}
					}
				case step.waitForPodsUnschedulable != nil:
					for _, podName := range step.waitForPodsUnschedulable {
						err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, wait.ForeverTestTimeout, false,
							testutils.PodUnschedulable(cs, ns, podName))
						if err != nil {
							t.Fatalf("Step %d: Failed to wait for pod %s to be unschedulable: %v", i, podName, err)
						}
					}
				case step.waitForPodsScheduled != nil:
					for _, podName := range step.waitForPodsScheduled {
						err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, wait.ForeverTestTimeout, false,
							testutils.PodScheduled(cs, ns, podName))
						if err != nil {
							t.Fatalf("Step %d: Failed to wait for pod %s to be scheduled: %v", i, podName, err)
						}
					}
				case step.waitForPodsRemoved != nil:
					for _, podName := range step.waitForPodsRemoved {
						err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, wait.ForeverTestTimeout, false,
							testutils.PodDeleted(testCtx.Ctx, cs, ns, podName))
						if err != nil {
							t.Fatalf("Step %d: Failed to wait for pod %s to be removed: %v", i, podName, err)
						}
					}
				case step.waitForAnyPodsScheduled != nil:
					err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, wait.ForeverTestTimeout, false,
						func(ctx context.Context) (bool, error) {
							scheduledCount := 0
							unschedulableCount := 0
							for _, pod := range step.waitForAnyPodsScheduled.pods {
								if ok, err := testutils.PodScheduled(cs, ns, pod.Name)(ctx); err != nil {
									return false, err
								} else if ok {
									scheduledCount++
									continue
								}
								if ok, err := testutils.PodUnschedulable(cs, ns, pod.Name)(ctx); err != nil {
									return false, err
								} else if ok {
									unschedulableCount++
								}
							}
							t.Logf("Step %d: Waiting for %d pods to be scheduled and %d to be unschedulable, got %d scheduled and %d unschedulable",
								i, step.waitForAnyPodsScheduled.numScheduled, step.waitForAnyPodsScheduled.numUnschedulable, scheduledCount, unschedulableCount)
							return scheduledCount == step.waitForAnyPodsScheduled.numScheduled && unschedulableCount == step.waitForAnyPodsScheduled.numUnschedulable, nil
						},
					)
					if err != nil {
						t.Fatalf("Step %d: Failed to wait for pods to be scheduled or unschedulable: %v", i, err)
					}
				}
			}
		})
	}
}

func TestWorkloadAwarePreemptionInvocation(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.GenericWorkload:         true,
		features.GangScheduling:          true,
		features.WorkloadAwarePreemption: true,
	})

	node := st.MakeNode().Name("node").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Obj()
	pg := st.MakePodGroup().Namespace("default").Name("pg1").TemplateRef("t1", "workload").MinCount(3).Obj()

	// Low priority pods taking up all resources
	lowPods := []*v1.Pod{
		st.MakePod().Namespace("default").Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").Priority(10).Obj(),
		st.MakePod().Namespace("default").Name("low-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").Priority(10).Obj(),
		st.MakePod().Namespace("default").Name("low-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").Priority(10).Obj(),
		st.MakePod().Namespace("default").Name("low-4").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").Priority(10).Obj(),
	}

	// High priority pods belonging to a group
	highPods := []*v1.Pod{
		st.MakePod().Namespace("default").Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").Priority(100).Obj(),
		st.MakePod().Namespace("default").Name("high-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").Priority(100).Obj(),
		st.MakePod().Namespace("default").Name("high-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").Priority(100).Obj(),
	}

	testCtx := testutils.InitTestSchedulerWithNS(t, "wap-inv",
		scheduler.WithPodMaxBackoffSeconds(0),
		scheduler.WithPodInitialBackoffSeconds(0))
	cs, ns := testCtx.ClientSet, testCtx.NS.Name

	_, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, node, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create node: %v", err)
	}

	// Intercept the PreemptionExecutor
	fwk := testCtx.Scheduler.Profiles[v1.DefaultSchedulerName]
	executor, ok := fwk.PreemptionExecutor().(*preemption.Executor)
	if !ok {
		t.Fatalf("PreemptionExecutor is not of type *preemption.Executor")
	}

	var wapCalled atomic.Int32
	var otherPreemptCalled atomic.Int32
	originalPreemptPod := executor.PreemptPod
	executor.PreemptPod = func(ctx context.Context, c preemption.Candidate, p preemption.ExecutorPreemptor, victim *v1.Pod, pluginName string) error {
		if pluginName == "workload-preemption" && p.Type() == "podgroup" {
			wapCalled.Add(1)
		} else {
			otherPreemptCalled.Add(1)
		}
		return originalPreemptPod(ctx, c, p, victim, pluginName)
	}
	defer func() {
		executor.PreemptPod = originalPreemptPod
	}()

	// 1. Create low priority pods
	for _, p := range lowPods {
		p.Namespace = ns
		if _, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, p, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create pod %s: %v", p.Name, err)
		}
	}

	// Wait for low priority pods to be scheduled
	for _, p := range lowPods {
		if err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, wait.ForeverTestTimeout, false,
			testutils.PodScheduled(cs, ns, p.Name)); err != nil {
			t.Fatalf("Failed to wait for pod %s to be scheduled: %v", p.Name, err)
		}
	}

	// 2. Create PodGroup
	pg.Namespace = ns
	if _, err := cs.SchedulingV1alpha2().PodGroups(ns).Create(testCtx.Ctx, pg, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create PodGroup: %v", err)
	}

	// 3. Create high priority pods
	for _, p := range highPods {
		p.Namespace = ns
		if _, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, p, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create pod %s: %v", p.Name, err)
		}
	}

	// 4. Verify that WorkloadAwarePreemption was called
	err = wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 10*time.Second, false, func(ctx context.Context) (bool, error) {
		return wapCalled.Load() > 0, nil
	})
	if err != nil {
		t.Errorf("WorkloadAwarePreemption was not called within timeout")
	}

	if otherPreemptCalled.Load() > 0 {
		t.Errorf("Default preemption was unexpectedly called %d times", otherPreemptCalled.Load())
	}

	t.Logf("WorkloadAwarePreemption was called %d times", wapCalled.Load())
}
