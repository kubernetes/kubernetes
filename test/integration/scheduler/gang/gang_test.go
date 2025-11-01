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

// podInUnschedulablePods checks if the given Pod is in the unschedulable pods pool within scheduling queue.
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
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.GenericWorkload: true,
		features.GangScheduling:  true,
	})

	testCtx := testutils.InitTestSchedulerWithNS(t, "gang-scheduling",
		// disable backoff
		scheduler.WithPodMaxBackoffSeconds(0),
		scheduler.WithPodInitialBackoffSeconds(0))

	workloadmanager.DefaultSchedulingTimeoutDuration = 5 * time.Second

	cs, ns := testCtx.ClientSet, testCtx.NS.Name

	node := st.MakeNode().Name("node").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Obj()

	workload := st.MakeWorkload().Name("workload").Namespace(ns).
		PodGroup(st.MakePodGroup().Name("pg").MinCount(3).Obj()).Obj()

	pod1 := st.MakePod().Name("pod1").Namespace(ns).Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").
		Workload(&v1.WorkloadReference{Name: "workload", PodGroup: "pg"}).Obj()
	pod2 := st.MakePod().Name("pod2").Namespace(ns).Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").
		Workload(&v1.WorkloadReference{Name: "workload", PodGroup: "pg"}).Obj()
	pod3 := st.MakePod().Name("pod3").Namespace(ns).Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").
		Workload(&v1.WorkloadReference{Name: "workload", PodGroup: "pg"}).Obj()
	gangPods := []*v1.Pod{pod1, pod2, pod3}

	otherPod := st.MakePod().Name("other-pod").Namespace(ns).Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").
		ZeroTerminationGracePeriod().Obj()

	_, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, node, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create node: %v", err)
	}

	// Create and schedule other pods that take up space on the node, preventing further gang scheduling.
	_, err = cs.CoreV1().Pods(ns).Create(testCtx.Ctx, otherPod, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create pod %s: %v", otherPod.Name, err)
	}
	err = testutils.WaitForPodToSchedule(testCtx.Ctx, cs, otherPod)
	if err != nil {
		t.Fatalf("Failed to wait for pod %s to be scheduled: %v", otherPod.Name, err)
	}

	// Create pods from the gang and wait until they are blocked in PreEnqueue (waiting in the unschedulable pods pool),
	// because their corresponding workload has not yet been created.
	for _, pod := range gangPods {
		_, err = cs.CoreV1().Pods(ns).Create(testCtx.Ctx, pod, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create pod %s: %v", pod.Name, err)
		}
	}
	for _, pod := range gangPods {
		err = wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, wait.ForeverTestTimeout, false,
			func(_ context.Context) (bool, error) {
				return podInUnschedulablePods(t, testCtx.Scheduler.SchedulingQueue, pod.Name), nil
			},
		)
		if err != nil {
			t.Fatalf("Failed to wait for pod %s to be in unschedulable pods pool: %v", pod.Name, err)
		}
	}

	// Create a workload that will unlock the gang pods.
	_, err = cs.SchedulingV1alpha1().Workloads(ns).Create(testCtx.Ctx, workload, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create workload: %v", err)
	}

	// Wait for the gang pods to be unschedulable - rejected by the timeout in WaitOnPermit,
	// because "other-pod" occupies space on the node for "pod3".
	for _, pod := range gangPods {
		err = testutils.WaitForPodUnschedulable(testCtx.Ctx, cs, pod)
		if err != nil {
			t.Fatalf("Failed to wait for pod %s to be unschedulable: %v", pod.Name, err)
		}
	}

	// Delete "other-pod", which causes "pod3" to retry its scheduling.
	err = cs.CoreV1().Pods(ns).Delete(testCtx.Ctx, otherPod.Name, metav1.DeleteOptions{})
	if err != nil {
		t.Fatalf("Failed to delete pod %s: %v", otherPod.Name, err)
	}

	// Wait for the entire gang to be scheduled.
	for _, pod := range gangPods {
		err = testutils.WaitForPodToSchedule(testCtx.Ctx, cs, pod)
		if err != nil {
			t.Fatalf("Failed to wait for pod %s to be scheduled: %v", pod.Name, err)
		}
	}
}
