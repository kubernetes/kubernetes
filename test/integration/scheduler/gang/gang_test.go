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

package gang

import (
	"context"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	schedulingapi "k8s.io/api/scheduling/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler"
	"k8s.io/kubernetes/pkg/scheduler/backend/queue"
	workloadmanager "k8s.io/kubernetes/pkg/scheduler/backend/workloadmanager"
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

func TestGangScheduling(t *testing.T) {
	node := st.MakeNode().Name("node").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Obj()

	workload := st.MakeWorkload().Name("workload").PodGroup(st.MakePodGroup().Name("pg").MinCount(3).Obj()).Obj()

	p1 := st.MakePod().Name("p1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").
		WorkloadRef(&v1.WorkloadReference{Name: "workload", PodGroup: "pg"}).Obj()
	p2 := st.MakePod().Name("p2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").
		WorkloadRef(&v1.WorkloadReference{Name: "workload", PodGroup: "pg"}).Obj()
	p3 := st.MakePod().Name("p3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").
		WorkloadRef(&v1.WorkloadReference{Name: "workload", PodGroup: "pg"}).Obj()
	blockerPod := st.MakePod().Name("blocker").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").
		ZeroTerminationGracePeriod().Obj()

	// step represents a single step in a test scenario.
	type step struct {
		name                         string
		createWorkload               *schedulingapi.Workload
		createPods                   []*v1.Pod
		deletePods                   []string
		waitForPodsGatedOnPreEnqueue []string
		waitForPodsUnschedulable     []string
		waitForPodsScheduled         []string
	}

	tests := []struct {
		name  string
		steps []step
	}{
		{
			name: "gang schedules when workload and resources are available",
			steps: []step{
				{
					name:           "Create the Workload object",
					createWorkload: workload,
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
					name:           "Create the Workload object",
					createWorkload: workload,
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
			name: "gang waits for workload, then for resources, then schedules",
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
					name:       "Create gang pods before Workload is created",
					createPods: []*v1.Pod{p1, p2, p3},
				},
				{
					name:                         "Verify pods are gated at PreEnqueue (no Workload object)",
					waitForPodsGatedOnPreEnqueue: []string{"p1", "p2", "p3"},
				},
				{
					name:           "Create the Workload to unblock PreEnqueue",
					createWorkload: workload,
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
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload: true,
				features.GangScheduling:  true,
			})

			workloadmanager.DefaultSchedulingTimeoutDuration = 5 * time.Second

			testCtx := testutils.InitTestSchedulerWithNS(t, "gang-scheduling",
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
				case step.createWorkload != nil:
					w := step.createWorkload.DeepCopy()
					w.Namespace = ns
					if _, err := cs.SchedulingV1alpha1().Workloads(ns).Create(testCtx.Ctx, w, metav1.CreateOptions{}); err != nil {
						t.Fatalf("Step %d: Failed to create workload %s: %v", i, w.Name, err)
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
				}
			}
		})
	}
}
