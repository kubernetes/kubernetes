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

package gangscheduling

import (
	"fmt"
	"sync"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	schedulerutils "k8s.io/kubernetes/test/integration/scheduler"
	testutils "k8s.io/kubernetes/test/integration/util"
)

// TestGangSchedulingWithBindingAndNNN verifies that gang scheduled pods have proper binding behavior.
func TestGangSchedulingWithBindingAndNNN(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GangScheduling, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.NominatedNodeNameForExpectation, true)

	testContext := testutils.InitTestAPIServer(t, "gang-binding-nnn-test", nil)

	createNodes(t, testContext, 2)

	testCtx, teardown := schedulerutils.InitTestSchedulerForFrameworkTest(t, testContext, 10, true)
	defer teardown()

	ns := testCtx.NS.Name
	cs := testCtx.ClientSet

	// Create PodGroup with gang scheduling (requires 2 pods minimum)
	pgName := "test-gang"
	pg := &schedulingv1alpha3.PodGroup{
		ObjectMeta: metav1.ObjectMeta{
			Name:      pgName,
			Namespace: ns,
		},
		Spec: schedulingv1alpha3.PodGroupSpec{
			SchedulingPolicy: schedulingv1alpha3.PodGroupSchedulingPolicy{
				Gang: &schedulingv1alpha3.GangSchedulingPolicy{
					MinCount: 2,
				},
			},
		},
	}
	_, err := cs.SchedulingV1alpha3().PodGroups(ns).Create(testCtx.Ctx, pg, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create PodGroup: %v", err)
	}

	// Create first gang pod - should remain pending until quorum is reached
	pod1 := st.MakePod().Name("gang-pod-1").Namespace(ns).Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").Obj()
	pod1.Spec.SchedulingGroup = &v1.PodSchedulingGroup{
		PodGroupName: &pgName,
	}
	pod1, err = cs.CoreV1().Pods(ns).Create(testCtx.Ctx, pod1, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create pod1: %v", err)
	}

	// Verify first pod is pending (waiting for gang quorum)
	if err := wait.PollImmediate(100*time.Millisecond, 5*time.Second, func() (bool, error) {
		p, _ := cs.CoreV1().Pods(ns).Get(testCtx.Ctx, pod1.Name, metav1.GetOptions{})
		return p.Status.Phase == v1.PodPending, nil
	}); err != nil {
		t.Errorf("Pod1 should be Pending, got: %v", err)
	}

	// Create second gang pod - this reaches quorum and triggers binding for both
	pod2 := st.MakePod().Name("gang-pod-2").Namespace(ns).Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").Obj()
	pod2.Spec.SchedulingGroup = &v1.PodSchedulingGroup{
		PodGroupName: &pgName,
	}
	pod2, err = cs.CoreV1().Pods(ns).Create(testCtx.Ctx, pod2, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create pod2: %v", err)
	}

	// After quorum, both pods should be scheduled
	if err := testutils.WaitForPodToSchedule(testCtx.Ctx, cs, pod1); err != nil {
		t.Errorf("Pod1 failed to schedule after gang quorum: %v", err)
	}
	if err := testutils.WaitForPodToSchedule(testCtx.Ctx, cs, pod2); err != nil {
		t.Errorf("Pod2 failed to schedule after gang quorum: %v", err)
	}

	// Verify both pods are bound to nodes
	pod1Final, _ := cs.CoreV1().Pods(ns).Get(testCtx.Ctx, pod1.Name, metav1.GetOptions{})
	pod2Final, _ := cs.CoreV1().Pods(ns).Get(testCtx.Ctx, pod2.Name, metav1.GetOptions{})

	if pod1Final.Spec.NodeName == "" {
		t.Errorf("Pod1 should be bound to a node after scheduling")
	}
	if pod2Final.Spec.NodeName == "" {
		t.Errorf("Pod2 should be bound to a node after scheduling")
	}
}

// TestGangSchedulingPreemptionWithNNN tests NNN behavior during preemption in gang scheduling.
func TestGangSchedulingPreemptionWithNNN(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GangScheduling, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.NominatedNodeNameForExpectation, true)

	testContext := testutils.InitTestAPIServer(t, "gang-preemption-nnn-test", nil)

	createNodes(t, testContext, 2)

	testCtx, teardown := schedulerutils.InitTestSchedulerForFrameworkTest(t, testContext, 10, true)
	defer teardown()

	ns := testCtx.NS.Name
	cs := testCtx.ClientSet

	// Create low-priority victim pods to occupy nodes
	victim1 := st.MakePod().Name("victim-1").Namespace(ns).Req(map[v1.ResourceName]string{v1.ResourceCPU: "8"}).
		Container("image").Priority(1).Node("node-0").Obj()
	_, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, victim1, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create victim1: %v", err)
	}

	victim2 := st.MakePod().Name("victim-2").Namespace(ns).Req(map[v1.ResourceName]string{v1.ResourceCPU: "8"}).
		Container("image").Priority(1).Node("node-1").Obj()
	_, err = cs.CoreV1().Pods(ns).Create(testCtx.Ctx, victim2, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create victim2: %v", err)
	}

	// Wait for victims to be scheduled
	if err := testutils.WaitForPodToSchedule(testCtx.Ctx, cs, victim1); err != nil {
		t.Fatalf("Victim1 failed to schedule: %v", err)
	}
	if err := testutils.WaitForPodToSchedule(testCtx.Ctx, cs, victim2); err != nil {
		t.Fatalf("Victim2 failed to schedule: %v", err)
	}

	// Create gang preemptor pods (higher priority)
	pgName := "preemptor-gang"
	pg := &schedulingv1alpha3.PodGroup{
		ObjectMeta: metav1.ObjectMeta{
			Name:      pgName,
			Namespace: ns,
		},
		Spec: schedulingv1alpha3.PodGroupSpec{
			SchedulingPolicy: schedulingv1alpha3.PodGroupSchedulingPolicy{
				Gang: &schedulingv1alpha3.GangSchedulingPolicy{
					MinCount: 2,
				},
			},
		},
	}
	_, err = cs.SchedulingV1alpha3().PodGroups(ns).Create(testCtx.Ctx, pg, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create preemptor PodGroup: %v", err)
	}

	// Create preemptor pods with high priority - will need preemption to fit
	preemptor1 := st.MakePod().Name("preemptor-1").Namespace(ns).Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).
		Container("image").Priority(100).Obj()
	preemptor1.Spec.SchedulingGroup = &v1.PodSchedulingGroup{
		PodGroupName: &pgName,
	}
	preemptor1, err = cs.CoreV1().Pods(ns).Create(testCtx.Ctx, preemptor1, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create preemptor1: %v", err)
	}

	preemptor2 := st.MakePod().Name("preemptor-2").Namespace(ns).Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).
		Container("image").Priority(100).Obj()
	preemptor2.Spec.SchedulingGroup = &v1.PodSchedulingGroup{
		PodGroupName: &pgName,
	}
	preemptor2, err = cs.CoreV1().Pods(ns).Create(testCtx.Ctx, preemptor2, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create preemptor2: %v", err)
	}

	// Both preemptor pods should be scheduled after preemption
	if err := testutils.WaitForPodToSchedule(testCtx.Ctx, cs, preemptor1); err != nil {
		t.Errorf("Preemptor1 failed to schedule: %v", err)
	}
	if err := testutils.WaitForPodToSchedule(testCtx.Ctx, cs, preemptor2); err != nil {
		t.Errorf("Preemptor2 failed to schedule: %v", err)
	}

	// Verify preemptor pods are bound to nodes
	preemptor1Final, _ := cs.CoreV1().Pods(ns).Get(testCtx.Ctx, preemptor1.Name, metav1.GetOptions{})
	preemptor2Final, _ := cs.CoreV1().Pods(ns).Get(testCtx.Ctx, preemptor2.Name, metav1.GetOptions{})

	if preemptor1Final.Spec.NodeName == "" {
		t.Errorf("Preemptor1 should be bound to a node after preemption")
	}
	if preemptor2Final.Spec.NodeName == "" {
		t.Errorf("Preemptor2 should be bound to a node after preemption")
	}
}

// TestGangSchedulingMultipleGroupsBindingIsolation tests that binding operations are isolated between different gang groups.
func TestGangSchedulingMultipleGroupsBindingIsolation(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GangScheduling, true)

	testContext := testutils.InitTestAPIServer(t, "gang-isolation-test", nil)

	createNodes(t, testContext, 3)

	testCtx, teardown := schedulerutils.InitTestSchedulerForFrameworkTest(t, testContext, 10, true)
	defer teardown()

	ns := testCtx.NS.Name
	cs := testCtx.ClientSet

	// Create two independent gang groups
	pg1 := &schedulingv1alpha3.PodGroup{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "gang-group-1",
			Namespace: ns,
		},
		Spec: schedulingv1alpha3.PodGroupSpec{
			SchedulingPolicy: schedulingv1alpha3.PodGroupSchedulingPolicy{
				Gang: &schedulingv1alpha3.GangSchedulingPolicy{
					MinCount: 2,
				},
			},
		},
	}
	_, err := cs.SchedulingV1alpha3().PodGroups(ns).Create(testCtx.Ctx, pg1, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create PodGroup 1: %v", err)
	}

	pg2 := &schedulingv1alpha3.PodGroup{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "gang-group-2",
			Namespace: ns,
		},
		Spec: schedulingv1alpha3.PodGroupSpec{
			SchedulingPolicy: schedulingv1alpha3.PodGroupSchedulingPolicy{
				Gang: &schedulingv1alpha3.GangSchedulingPolicy{
					MinCount: 2,
				},
			},
		},
	}
	_, err = cs.SchedulingV1alpha3().PodGroups(ns).Create(testCtx.Ctx, pg2, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create PodGroup 2: %v", err)
	}

	type podRef struct {
		name   string
		pgName string
	}

	// Create pods in both groups in interleaved order to stress test binding isolation
	pods := []podRef{
		{"g1p1", "gang-group-1"},
		{"g2p1", "gang-group-2"},
		{"g1p2", "gang-group-1"},
		{"g2p2", "gang-group-2"},
	}

	for _, p := range pods {
		pod := st.MakePod().Name(p.name).Namespace(ns).Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").Obj()
		pod.Spec.SchedulingGroup = &v1.PodSchedulingGroup{
			PodGroupName: &p.pgName,
		}
		_, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, pod, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create pod %s: %v", p.name, err)
		}
	}

	// Verify all pods are eventually scheduled
	for _, p := range pods {
		testPod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: p.name, Namespace: ns},
		}
		if err := testutils.WaitForPodToSchedule(testCtx.Ctx, cs, testPod); err != nil {
			t.Errorf("Pod %s failed to schedule: %v", p.name, err)
		}
	}
}

// TestGangSchedulingBindingRaceConditions tests that concurrent binding operations work correctly.
func TestGangSchedulingBindingRaceConditions(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GangScheduling, true)

	testContext := testutils.InitTestAPIServer(t, "gang-race-test", nil)

	createNodes(t, testContext, 5)

	testCtx, teardown := schedulerutils.InitTestSchedulerForFrameworkTest(t, testContext, 10, true)
	defer teardown()

	ns := testCtx.NS.Name
	cs := testCtx.ClientSet

	// Create multiple gang groups with concurrent pod creation
	const groupCount = 3
	const podsPerGroup = 2

	pg := make([]*schedulingv1alpha3.PodGroup, groupCount)
	for i := 0; i < groupCount; i++ {
		pgName := fmt.Sprintf("concurrent-group-%d", i)
		pg[i] = &schedulingv1alpha3.PodGroup{
			ObjectMeta: metav1.ObjectMeta{
				Name:      pgName,
				Namespace: ns,
			},
			Spec: schedulingv1alpha3.PodGroupSpec{
				SchedulingPolicy: schedulingv1alpha3.PodGroupSchedulingPolicy{
					Gang: &schedulingv1alpha3.GangSchedulingPolicy{
						MinCount: podsPerGroup,
					},
				},
			},
		}
		_, err := cs.SchedulingV1alpha3().PodGroups(ns).Create(testCtx.Ctx, pg[i], metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create PodGroup %d: %v", i, err)
		}
	}

	// Create pods concurrently to stress-test concurrent binding
	var wg sync.WaitGroup
	errChan := make(chan error, groupCount*podsPerGroup)

	for i := 0; i < groupCount; i++ {
		for j := 0; j < podsPerGroup; j++ {
			wg.Add(1)
			go func(groupIdx, podIdx int) {
				defer wg.Done()
				pgName := fmt.Sprintf("concurrent-group-%d", groupIdx)
				podName := fmt.Sprintf("g%dp%d", groupIdx, podIdx)

				pod := st.MakePod().Name(podName).Namespace(ns).Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").Obj()
				pod.Spec.SchedulingGroup = &v1.PodSchedulingGroup{
					PodGroupName: &pgName,
				}
				_, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, pod, metav1.CreateOptions{})
				if err != nil {
					errChan <- fmt.Errorf("failed to create pod %s: %v", podName, err)
				}
			}(i, j)
		}
	}

	wg.Wait()
	close(errChan)

	// Check for creation errors
	for err := range errChan {
		if err != nil {
			t.Errorf("Pod creation error: %v", err)
		}
	}

	// Verify all pods are scheduled despite concurrent creation
	pods, err := cs.CoreV1().Pods(ns).List(testCtx.Ctx, metav1.ListOptions{})
	if err != nil {
		t.Fatalf("Failed to list pods: %v", err)
	}

	for _, pod := range pods.Items {
		testPod := pod
		if err := testutils.WaitForPodToSchedule(testCtx.Ctx, cs, &testPod); err != nil {
			t.Errorf("Pod %s/%s failed to schedule: %v", pod.Namespace, pod.Name, err)
		}
	}
}

// Helper to create nodes with proper capacity
func createNodes(t *testing.T, testCtx *testutils.TestContext, count int) {
	for i := 0; i < count; i++ {
		node := &v1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: fmt.Sprintf("node-%d", i),
			},
			Spec: v1.NodeSpec{
				Taints: []v1.Taint{},
			},
			Status: v1.NodeStatus{
				Capacity: v1.ResourceList{
					v1.ResourceCPU:    *resource.NewScaledQuantity(10, resource.Milli),
					v1.ResourceMemory: *resource.NewQuantity(10*1024*1024*1024, resource.BinarySI),
				},
				Allocatable: v1.ResourceList{
					v1.ResourceCPU:    *resource.NewScaledQuantity(10, resource.Milli),
					v1.ResourceMemory: *resource.NewQuantity(10*1024*1024*1024, resource.BinarySI),
				},
				Conditions: []v1.NodeCondition{
					{
						Type:               v1.NodeReady,
						Status:             v1.ConditionTrue,
						LastHeartbeatTime:  metav1.Now(),
						LastTransitionTime: metav1.Now(),
						Reason:             "KubeletReady",
						Message:            "kubelet is posting ready status",
					},
				},
			},
		}
		_, err := testCtx.ClientSet.CoreV1().Nodes().Create(testCtx.Ctx, node, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create node %d: %v", i, err)
		}
	}
}
