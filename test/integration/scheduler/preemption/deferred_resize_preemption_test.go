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
	policyv1 "k8s.io/api/policy/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	configv1 "k8s.io/kube-scheduler/config/v1"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler"
	configtesting "k8s.io/kubernetes/pkg/scheduler/apis/config/testing"
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
	})

	cfg := configtesting.V1ToInternalWithDefaults(t, configv1.KubeSchedulerConfiguration{
		Profiles: []configv1.KubeSchedulerProfile{{
			SchedulerName: new(v1.DefaultSchedulerName),
		}},
	})

	tests := []struct {
		name                 string
		nodeCapacityCPU      string
		nodeCapacityMem      string
		nodePreemptionPolicy *v1.NodePodPreemptionPolicy
		existingPods         []*v1.Pod
		preemptorConfig      *testutils.PausePodConfig
		expectEvictedNames   []string
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
			expectEvictedNames: nil,
		},
		{
			name:            "parking strategy when deferred resize fits on node",
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
			expectEvictedNames: nil,
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
			expectEvictedNames: nil,
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
			nodeRes := map[v1.ResourceName]string{
				v1.ResourcePods:   "32",
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
			defer func() {
				_ = cs.CoreV1().Nodes().Delete(testCtx.Ctx, nodeName, metav1.DeleteOptions{})
			}()

			// Create and run existing pods (if any) on nodeName
			var pods []*v1.Pod
			for _, p := range tt.existingPods {
				p.Namespace = testCtx.NS.Name
				p.Spec.NodeName = nodeName
				runningPod, err := runPausePod(cs, p)
				if err != nil {
					t.Fatalf("Failed running pause pod %v: %v", p.Name, err)
				}
				pods = append(pods, runningPod)
			}

			// Create preemptor/resizing pod already scheduled to nodeName
			tt.preemptorConfig.NodeName = nodeName
			tt.preemptorConfig.Namespace = testCtx.NS.Name
			preemptorPod := initPausePod(tt.preemptorConfig)
			preemptorPod, err := cs.CoreV1().Pods(testCtx.NS.Name).Create(testCtx.Ctx, preemptorPod, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("Failed to create preemptor pod: %v", err)
			}

			// Update status of preemptor pod to simulate it running and being deferred.
			// Allocated CPU/Mem is set to 100m/100, while Spec request is 300m/100.
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
			preemptorPod, err = cs.CoreV1().Pods(testCtx.NS.Name).UpdateStatus(testCtx.Ctx, preemptorPod, metav1.UpdateOptions{})
			if err != nil {
				t.Fatalf("Failed to update status of preemptor pod: %v", err)
			}

			// Wait for expected evictions (if any) or timeout
			if len(tt.expectEvictedNames) > 0 {
				for _, name := range tt.expectEvictedNames {
					err = wait.PollUntilContextTimeout(testCtx.Ctx, 50*time.Millisecond, 10*time.Second, false,
						podIsGettingEvicted(cs, testCtx.NS.Name, name))
					if err != nil {
						t.Errorf("Expected pod %q to be evicted/deleted, but it was not", name)
					}
				}
			} else {
				// Wait to confirm NO evictions happen
				time.Sleep(500 * time.Millisecond)
				for _, p := range pods {
					livePod, err := cs.CoreV1().Pods(testCtx.NS.Name).Get(testCtx.Ctx, p.Name, metav1.GetOptions{})
					if err == nil && livePod.DeletionTimestamp != nil {
						t.Errorf("Expected pod %q NOT to be evicted/deleted, but its DeletionTimestamp is set", p.Name)
					}
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

			// Verify the pod remains parked in the scheduler queue
			queue := testCtx.Scheduler.SchedulingQueue
			_, found := queue.GetPod(testCtx.Ctx, preemptorPod.Name, preemptorPod.Namespace, nil)
			if !found {
				t.Errorf("Expected preemptor pod to be found in scheduling queue")
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

			// Cleanup
			pods = append(pods, preemptorPod)
			testutils.CleanupPods(testCtx.Ctx, cs, t, pods)
		})
	}
}

func setUpPreemptionTestWithContext(t *testing.T, testCtx *testutils.TestContext, idx int, nodeCPUCapacity, otherPodCPURequest, deferredPodCPURequest, deferredPodCPUAllocated string, disableResizePreemption bool) (*v1.Pod, *v1.Pod, *v1.Pod) {
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

	// Create node2 (irrelevant node)
	nodeObject2 := st.MakeNode().Name(nodeName2).Capacity(map[v1.ResourceName]string{
		v1.ResourcePods: "32",
		v1.ResourceCPU:  "500m",
	}).Obj()
	if _, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, nodeObject2, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create node2: %v", err)
	}

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
				err := wait.PollUntilContextTimeout(testCtx.Ctx, 50*time.Millisecond, 5*time.Second, false, func(ctx context.Context) (bool, error) {
					qPod, found := queue.GetPod(ctx, pod.Name, pod.Namespace, nil)
					return found && qPod.Attempts > initialAttempts, nil
				})
				if err != nil {
					t.Fatalf("Expected queue attempts to increment after trigger")
				}
			} else {
				time.Sleep(300 * time.Millisecond)
				qPod, found := queue.GetPod(testCtx.Ctx, pod.Name, pod.Namespace, nil)
				if found && qPod.Attempts > initialAttempts {
					t.Fatalf("Expected attempts not to increment, but went from %d to %d", initialAttempts, qPod.Attempts)
				}
			}

			testutils.CleanupPods(testCtx.Ctx, cs, t, []*v1.Pod{pod, other, irrelevant})
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

	t.Run("disabling node preemption policy does not wake queue", func(t *testing.T) {
		pod, other, irrelevant := setUpPreemptionTestWithContext(t, testCtx, 0, "200m", "200m", "100m", "50m", false)

		cs := testCtx.ClientSet
		queue := testCtx.Scheduler.SchedulingQueue

		queuedPod, found := queue.GetPod(testCtx.Ctx, pod.Name, pod.Namespace, nil)
		if !found {
			t.Fatalf("Expected pod to be in queue initially")
		}
		initialAttempts := queuedPod.Attempts

		// Disable preemption on assigned node
		nodeName := pod.Spec.NodeName
		n, err := cs.CoreV1().Nodes().Get(testCtx.Ctx, nodeName, metav1.GetOptions{})
		if err != nil {
			t.Fatalf("Failed to get node: %v", err)
		}
		n.Spec.PodPreemptionPolicy = &v1.NodePodPreemptionPolicy{
			DisableResizePreemption: []string{"test-policy"},
		}
		if _, err := cs.CoreV1().Nodes().Update(testCtx.Ctx, n, metav1.UpdateOptions{}); err != nil {
			t.Fatalf("Failed to update node to disable preemption: %v", err)
		}

		// Verify pod attempts do not increment (QHint skipped)
		time.Sleep(300 * time.Millisecond)
		qPod, found := queue.GetPod(testCtx.Ctx, pod.Name, pod.Namespace, nil)
		if !found || qPod.Attempts > initialAttempts {
			t.Fatalf("Expected attempts not to increment, but went from %d to %d (or pod was lost)", initialAttempts, qPod.Attempts)
		}

		testutils.CleanupPods(testCtx.Ctx, cs, t, []*v1.Pod{pod, other, irrelevant})
	})

	t.Run("enabling node preemption policy wakes queue", func(t *testing.T) {
		pod, other, irrelevant := setUpPreemptionTestWithContext(t, testCtx, 1, "200m", "200m", "100m", "50m", true)

		cs := testCtx.ClientSet
		queue := testCtx.Scheduler.SchedulingQueue

		queuedPod, found := queue.GetPod(testCtx.Ctx, pod.Name, pod.Namespace, nil)
		if !found {
			t.Fatalf("Expected pod to still be in queue")
		}
		initialAttempts := queuedPod.Attempts

		// Now re-enable preemption on node (clear DisableResizePreemption)
		nodeName := pod.Spec.NodeName
		n, err := cs.CoreV1().Nodes().Get(testCtx.Ctx, nodeName, metav1.GetOptions{})
		if err != nil {
			t.Fatalf("Failed to get node: %v", err)
		}
		n.Spec.PodPreemptionPolicy = nil
		if _, err := cs.CoreV1().Nodes().Update(testCtx.Ctx, n, metav1.UpdateOptions{}); err != nil {
			t.Fatalf("Failed to update node to enable preemption: %v", err)
		}

		// Verify pod is retried (attempts incremented)
		err = wait.PollUntilContextTimeout(testCtx.Ctx, 50*time.Millisecond, 5*time.Second, false, func(ctx context.Context) (bool, error) {
			qPod, found := queue.GetPod(ctx, pod.Name, pod.Namespace, nil)
			return found && qPod.Attempts > initialAttempts, nil
		})
		if err != nil {
			t.Fatalf("Expected attempts to increment after re-enabling node preemption policy")
		}

		testutils.CleanupPods(testCtx.Ctx, cs, t, []*v1.Pod{pod, other, irrelevant})
	})

	t.Run("irrelevant node preemption policy change ignores queue", func(t *testing.T) {
		pod, other, irrelevant := setUpPreemptionTestWithContext(t, testCtx, 2, "200m", "200m", "100m", "50m", false)

		cs := testCtx.ClientSet
		queue := testCtx.Scheduler.SchedulingQueue

		queuedPod, found := queue.GetPod(testCtx.Ctx, pod.Name, pod.Namespace, nil)
		if !found {
			t.Fatalf("Expected pod to be in queue initially")
		}
		initialAttempts := queuedPod.Attempts

		// Update irrelevant node2
		nodeName2 := fmt.Sprintf("hint-node2-%d", 2)
		n, err := cs.CoreV1().Nodes().Get(testCtx.Ctx, nodeName2, metav1.GetOptions{})
		if err != nil {
			t.Fatalf("Failed to get node2: %v", err)
		}
		n.Spec.PodPreemptionPolicy = &v1.NodePodPreemptionPolicy{
			DisableResizePreemption: []string{"test-policy"},
		}
		if _, err := cs.CoreV1().Nodes().Update(testCtx.Ctx, n, metav1.UpdateOptions{}); err != nil {
			t.Fatalf("Failed to update node2: %v", err)
		}

		time.Sleep(300 * time.Millisecond)
		qPod, found := queue.GetPod(testCtx.Ctx, pod.Name, pod.Namespace, nil)
		if !found || qPod.Attempts > initialAttempts {
			t.Fatalf("Expected attempts not to increment and pod to remain in queue for irrelevant node update")
		}

		testutils.CleanupPods(testCtx.Ctx, cs, t, []*v1.Pod{pod, other, irrelevant})
	})
}

func TestDeferredResizeQueueingHandlers(t *testing.T) {
	t.Run("AddPod handler enqueues existing deferred pod on scheduler startup", func(t *testing.T) {
		featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
			features.InPlacePodVerticalScaling:                    true,
			features.InPlacePodVerticalScalingSchedulerPreemption: true,
		})

		cfg := configtesting.V1ToInternalWithDefaults(t, configv1.KubeSchedulerConfiguration{
			Profiles: []configv1.KubeSchedulerProfile{{
				SchedulerName: new(v1.DefaultSchedulerName),
			}},
		})

		// Start ONLY the API Server (no scheduler yet!)
		testCtx := testutils.InitTestAPIServer(t, "deferred-handlers-test", nil)

		cs := testCtx.ClientSet

		// Create node
		nodeObject := st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{
			v1.ResourcePods: "32",
			v1.ResourceCPU:  "200m",
		}).Obj()
		if _, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, nodeObject, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create node: %v", err)
		}

		// Create and status-update pod to be deferred resize
		pod := initPausePod(&testutils.PausePodConfig{
			Name:     "deferred-pod",
			NodeName: "node1",
			Priority: &asyncframework.LowPriority,
			Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
				v1.ResourceCPU: *resource.NewMilliQuantity(100, resource.DecimalSI)},
			},
		})
		pod.Namespace = testCtx.NS.Name
		pod, err := cs.CoreV1().Pods(testCtx.NS.Name).Create(testCtx.Ctx, pod, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create pod: %v", err)
		}

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
					v1.ResourceCPU: *resource.NewMilliQuantity(50, resource.DecimalSI),
				},
				Resources: &v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceCPU: *resource.NewMilliQuantity(50, resource.DecimalSI),
					},
				},
			},
		}
		pod, err = cs.CoreV1().Pods(testCtx.NS.Name).UpdateStatus(testCtx.Ctx, pod, metav1.UpdateOptions{})
		if err != nil {
			t.Fatalf("Failed to update status: %v", err)
		}

		// Initialize Scheduler (which registers handlers and lists existing pods)
		testCtx = testutils.InitTestSchedulerWithOptions(t, testCtx, 0, scheduler.WithProfiles(cfg.Profiles...))
		defer testCtx.SchedulerCloseFn()
		testutils.SyncSchedulerInformerFactory(testCtx)

		// Verify the pod was immediately enqueued on startup by the AddPod handler!
		queue := testCtx.Scheduler.SchedulingQueue
		var found bool
		err = wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 2*time.Second, true, func(ctx context.Context) (bool, error) {
			_, found = queue.GetPod(ctx, pod.Name, pod.Namespace, nil)
			return found, nil
		})
		if err != nil || !found {
			t.Fatalf("Expected pod to be queued automatically by AddPod handler on scheduler startup, err: %v, found: %t", err, found)
		}

		testutils.CleanupPods(testCtx.Ctx, cs, t, []*v1.Pod{pod})
	})

	t.Run("UpdatePod handler enqueues pod when it transitions to deferred resize", func(t *testing.T) {
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
			testutils.InitTestAPIServer(t, "deferred-handlers-test", nil),
			0,
			scheduler.WithProfiles(cfg.Profiles...),
		)
		defer testCtx.SchedulerCloseFn()
		testutils.SyncSchedulerInformerFactory(testCtx)

		// Note: we don't start the scheduling loop (Scheduler.Run), only informers & queue!
		cs := testCtx.ClientSet
		queue := testCtx.Scheduler.SchedulingQueue

		// Create node
		nodeObject := st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{
			v1.ResourcePods: "32",
			v1.ResourceCPU:  "200m",
		}).Obj()
		if _, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, nodeObject, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create node: %v", err)
		}

		// Create a normal running pod assigned to node1 (no deferred condition)
		pod := initPausePod(&testutils.PausePodConfig{
			Name:     "deferred-pod",
			NodeName: "node1",
			Priority: &asyncframework.LowPriority,
			Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
				v1.ResourceCPU: *resource.NewMilliQuantity(100, resource.DecimalSI)},
			},
		})
		pod.Namespace = testCtx.NS.Name
		pod, err := cs.CoreV1().Pods(testCtx.NS.Name).Create(testCtx.Ctx, pod, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create pod: %v", err)
		}

		pod.Status.Phase = v1.PodRunning
		pod.Status.Conditions = []v1.PodCondition{
			{
				Type:   v1.PodScheduled,
				Status: v1.ConditionTrue,
			},
		}
		pod, err = cs.CoreV1().Pods(testCtx.NS.Name).UpdateStatus(testCtx.Ctx, pod, metav1.UpdateOptions{})
		if err != nil {
			t.Fatalf("Failed to update status: %v", err)
		}

		// Verify the pod is NOT in the scheduling queue
		time.Sleep(300 * time.Millisecond)
		if _, found := queue.GetPod(testCtx.Ctx, pod.Name, pod.Namespace, nil); found {
			t.Fatalf("Expected pod to not be in scheduling queue initially")
		}

		// Now update status to add the deferred condition
		pod, err = cs.CoreV1().Pods(testCtx.NS.Name).Get(testCtx.Ctx, pod.Name, metav1.GetOptions{})
		if err != nil {
			t.Fatalf("Failed to get pod: %v", err)
		}
		pod.Status.Conditions = append(pod.Status.Conditions, v1.PodCondition{
			Type:   v1.PodResizePending,
			Status: v1.ConditionTrue,
			Reason: v1.PodReasonDeferred,
		})
		pod.Status.ContainerStatuses = []v1.ContainerStatus{
			{
				Name: pod.Name,
				AllocatedResources: v1.ResourceList{
					v1.ResourceCPU: *resource.NewMilliQuantity(50, resource.DecimalSI),
				},
				Resources: &v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceCPU: *resource.NewMilliQuantity(50, resource.DecimalSI),
					},
				},
			},
		}
		pod, err = cs.CoreV1().Pods(testCtx.NS.Name).UpdateStatus(testCtx.Ctx, pod, metav1.UpdateOptions{})
		if err != nil {
			t.Fatalf("Failed to transition status: %v", err)
		}

		// Verify the pod was enqueued automatically by the UpdatePod event handler!
		var found bool
		err = wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 2*time.Second, true, func(ctx context.Context) (bool, error) {
			_, found = queue.GetPod(ctx, pod.Name, pod.Namespace, nil)
			return found, nil
		})
		if err != nil || !found {
			t.Fatalf("Expected pod to be enqueued automatically by UpdatePod handler on deferred status transition")
		}

		testutils.CleanupPods(testCtx.Ctx, cs, t, []*v1.Pod{pod})
	})

	t.Run("UpdatePod handler removes pod from queue when deferred condition is cleared", func(t *testing.T) {
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
			testutils.InitTestAPIServer(t, "deferred-handlers-test", nil),
			0,
			scheduler.WithProfiles(cfg.Profiles...),
		)
		defer testCtx.SchedulerCloseFn()
		testutils.SyncSchedulerInformerFactory(testCtx)

		cs := testCtx.ClientSet
		queue := testCtx.Scheduler.SchedulingQueue

		nodeObject := st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{
			v1.ResourcePods: "32",
			v1.ResourceCPU:  "200m",
		}).Obj()
		if _, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, nodeObject, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create node: %v", err)
		}

		pod := initPausePod(&testutils.PausePodConfig{
			Name:     "deferred-pod",
			NodeName: "node1",
			Priority: &asyncframework.LowPriority,
			Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
				v1.ResourceCPU: *resource.NewMilliQuantity(100, resource.DecimalSI)},
			},
		})
		pod.Namespace = testCtx.NS.Name
		pod, err := cs.CoreV1().Pods(testCtx.NS.Name).Create(testCtx.Ctx, pod, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create pod: %v", err)
		}

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
		pod, err = cs.CoreV1().Pods(testCtx.NS.Name).UpdateStatus(testCtx.Ctx, pod, metav1.UpdateOptions{})
		if err != nil {
			t.Fatalf("Failed to update status: %v", err)
		}

		err = wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 2*time.Second, true, func(ctx context.Context) (bool, error) {
			_, found := queue.GetPod(ctx, pod.Name, pod.Namespace, nil)
			return found, nil
		})
		if err != nil {
			t.Fatalf("Expected pod to be in queue after transitioning to deferred")
		}

		pod, err = cs.CoreV1().Pods(testCtx.NS.Name).Get(testCtx.Ctx, pod.Name, metav1.GetOptions{})
		if err != nil {
			t.Fatalf("Failed to get pod: %v", err)
		}
		pod.Status.Conditions = []v1.PodCondition{
			{
				Type:   v1.PodScheduled,
				Status: v1.ConditionTrue,
			},
		}
		if _, err := cs.CoreV1().Pods(testCtx.NS.Name).UpdateStatus(testCtx.Ctx, pod, metav1.UpdateOptions{}); err != nil {
			t.Fatalf("Failed to update status clearing deferred condition: %v", err)
		}

		err = wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 2*time.Second, true, func(ctx context.Context) (bool, error) {
			_, found := queue.GetPod(ctx, pod.Name, pod.Namespace, nil)
			return !found, nil
		})
		if err != nil {
			t.Fatalf("Expected pod to be removed from queue after deferred condition cleared")
		}

		testutils.CleanupPods(testCtx.Ctx, cs, t, []*v1.Pod{pod})
	})

	t.Run("DeletePod handler removes pod from both cache and scheduling queue when deleted", func(t *testing.T) {
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
			testutils.InitTestAPIServer(t, "def-del", nil),
			0,
			scheduler.WithProfiles(cfg.Profiles...),
		)
		defer testCtx.SchedulerCloseFn()
		testutils.SyncSchedulerInformerFactory(testCtx)

		cs := testCtx.ClientSet
		queue := testCtx.Scheduler.SchedulingQueue
		cache := testCtx.Scheduler.Cache

		// Create node
		nodeObject := st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{
			v1.ResourcePods: "32",
			v1.ResourceCPU:  "200m",
		}).Obj()
		if _, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, nodeObject, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create node: %v", err)
		}

		// Create a deferred resize pod assigned to node1
		pod := initPausePod(&testutils.PausePodConfig{
			Name:     "deferred-pod-del",
			NodeName: "node1",
			Priority: &asyncframework.LowPriority,
			Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
				v1.ResourceCPU: *resource.NewMilliQuantity(100, resource.DecimalSI)},
			},
		})
		pod.Namespace = testCtx.NS.Name
		pod, err := cs.CoreV1().Pods(testCtx.NS.Name).Create(testCtx.Ctx, pod, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create pod: %v", err)
		}

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
					v1.ResourceCPU: *resource.NewMilliQuantity(50, resource.DecimalSI),
				},
				Resources: &v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceCPU: *resource.NewMilliQuantity(50, resource.DecimalSI),
					},
				},
			},
		}
		pod, err = cs.CoreV1().Pods(testCtx.NS.Name).UpdateStatus(testCtx.Ctx, pod, metav1.UpdateOptions{})
		if err != nil {
			t.Fatalf("Failed to update status: %v", err)
		}

		// Wait for pod to be enqueued and in cache
		err = wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 5*time.Second, true, func(ctx context.Context) (bool, error) {
			_, foundInQueue := queue.GetPod(ctx, pod.Name, pod.Namespace, nil)
			_, errCache := cache.GetPod(pod)
			return foundInQueue && errCache == nil, nil
		})
		if err != nil {
			t.Fatalf("Expected pod to be in both cache and queue initially: %v", err)
		}

		// Delete the pod
		if err := cs.CoreV1().Pods(testCtx.NS.Name).Delete(testCtx.Ctx, pod.Name, metav1.DeleteOptions{GracePeriodSeconds: ptr.To[int64](0)}); err != nil {
			t.Fatalf("Failed to delete pod: %v", err)
		}

		// Verify the pod is removed from both queue and cache by DeletePod event handler
		err = wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 5*time.Second, true, func(ctx context.Context) (bool, error) {
			_, foundInQueue := queue.GetPod(ctx, pod.Name, pod.Namespace, nil)
			_, errCache := cache.GetPod(pod)
			return !foundInQueue && errCache != nil, nil
		})
		if err != nil {
			t.Fatalf("Expected pod to be removed from both cache and queue after deletion: %v", err)
		}
	})
}

func updatePodToDeferredResize(ctx context.Context, cs clientset.Interface, pod *v1.Pod, allocatedCPU, allocatedMem string) (*v1.Pod, error) {
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
				v1.ResourceCPU:    resource.MustParse(allocatedCPU),
				v1.ResourceMemory: resource.MustParse(allocatedMem),
			},
			Resources: &v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse(allocatedCPU),
					v1.ResourceMemory: resource.MustParse(allocatedMem),
				},
			},
		},
	}
	return cs.CoreV1().Pods(pod.Namespace).UpdateStatus(ctx, pod, metav1.UpdateOptions{})
}

func addPodReadyCondition(pod *v1.Pod) {
	pod.Status.Conditions = append(pod.Status.Conditions, v1.PodCondition{
		Type:   v1.PodReady,
		Status: v1.ConditionTrue,
	})
}

func TestDeferredResizePodPreemption_PDBCompliance(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.InPlacePodVerticalScaling:                    true,
		features.InPlacePodVerticalScalingSchedulerPreemption: true,
	})

	cfg := configtesting.V1ToInternalWithDefaults(t, configv1.KubeSchedulerConfiguration{
		Profiles: []configv1.KubeSchedulerProfile{{
			SchedulerName: new(v1.DefaultSchedulerName),
		}},
	})

	t.Run("prefer non-PDB violating pod over lower priority PDB protected pod", func(t *testing.T) {
		testCtx := testutils.InitTestSchedulerWithOptions(t,
			testutils.InitTestAPIServer(t, "def-resize-pdb-1", nil),
			0,
			scheduler.WithProfiles(cfg.Profiles...))
		testutils.SyncSchedulerInformerFactory(testCtx)
		go testCtx.Scheduler.Run(testCtx.SchedulerCtx)
		defer testCtx.SchedulerCloseFn()

		testutils.InitDisruptionController(t, testCtx)
		cs := testCtx.ClientSet
		ns := testCtx.NS.Name

		nodeName := "pdb-node-1"
		node := st.MakeNode().Name(nodeName).Capacity(map[v1.ResourceName]string{
			v1.ResourcePods:   "32",
			v1.ResourceCPU:    "400m",
			v1.ResourceMemory: "400Mi",
		}).Label("node", nodeName).Obj()
		if _, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, node, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create node: %v", err)
		}
		defer func() {
			_ = cs.CoreV1().Nodes().Delete(testCtx.Ctx, nodeName, metav1.DeleteOptions{})
		}()

		// victim-pdb: priority Low (0), requests 100m CPU, protected by PDB minAvailable=1
		victimPDB := initPausePod(&testutils.PausePodConfig{
			Name:      "victim-pdb",
			Namespace: ns,
			Priority:  &asyncframework.LowPriority,
			Labels:    map[string]string{"app": "pdb-app"},
			Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
				v1.ResourceCPU:    *resource.NewMilliQuantity(100, resource.DecimalSI),
				v1.ResourceMemory: *resource.NewQuantity(100, resource.BinarySI),
			}},
		})
		victimPDB.Spec.NodeName = nodeName
		victimPDB, err := runPausePod(cs, victimPDB)
		if err != nil {
			t.Fatalf("Failed to run victim-pdb: %v", err)
		}
		addPodReadyCondition(victimPDB)
		if _, err := cs.CoreV1().Pods(ns).UpdateStatus(testCtx.Ctx, victimPDB, metav1.UpdateOptions{}); err != nil {
			t.Fatalf("Failed to update status for victim-pdb: %v", err)
		}

		// victim-non-pdb: priority Mid (200), requests 200m CPU, not covered by PDB
		victimNonPDB := initPausePod(&testutils.PausePodConfig{
			Name:      "victim-non-pdb",
			Namespace: ns,
			Priority:  &asyncframework.MediumPriority,
			Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
				v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
				v1.ResourceMemory: *resource.NewQuantity(100, resource.BinarySI),
			}},
		})
		victimNonPDB.Spec.NodeName = nodeName
		victimNonPDB, err = runPausePod(cs, victimNonPDB)
		if err != nil {
			t.Fatalf("Failed to run victim-non-pdb: %v", err)
		}
		addPodReadyCondition(victimNonPDB)
		if _, err := cs.CoreV1().Pods(ns).UpdateStatus(testCtx.Ctx, victimNonPDB, metav1.UpdateOptions{}); err != nil {
			t.Fatalf("Failed to update status for victim-non-pdb: %v", err)
		}

		if err := testutils.WaitCachedPodsStable(testCtx, []*v1.Pod{victimPDB, victimNonPDB}); err != nil {
			t.Fatalf("Pods not stable in cache: %v", err)
		}

		// Create PDB protecting victim-pdb (minAvailable: 1)
		minAvailable := intstr.FromInt32(1)
		pdb := &policyv1.PodDisruptionBudget{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "pdb-protect",
				Namespace: ns,
			},
			Spec: policyv1.PodDisruptionBudgetSpec{
				MinAvailable: &minAvailable,
				Selector:     &metav1.LabelSelector{MatchLabels: map[string]string{"app": "pdb-app"}},
			},
		}
		if _, err := cs.PolicyV1().PodDisruptionBudgets(ns).Create(testCtx.Ctx, pdb, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create PDB: %v", err)
		}

		if err := testutils.WaitForPDBsStable(testCtx, []*policyv1.PodDisruptionBudget{pdb}, []int32{1}); err != nil {
			t.Fatalf("PDB not stable: %v", err)
		}

		// Preemptor pod requesting resize from 100m to 300m CPU (needs 200m additional CPU)
		preemptorPod := initPausePod(&testutils.PausePodConfig{
			Name:      "preemptor-pod",
			Namespace: ns,
			Priority:  &asyncframework.HighPriority,
			Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
				v1.ResourceCPU:    *resource.NewMilliQuantity(300, resource.DecimalSI),
				v1.ResourceMemory: *resource.NewQuantity(100, resource.BinarySI),
			}},
		})
		preemptorPod.Spec.NodeName = nodeName
		preemptorPod, err = cs.CoreV1().Pods(ns).Create(testCtx.Ctx, preemptorPod, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create preemptor pod: %v", err)
		}
		preemptorPod, err = updatePodToDeferredResize(testCtx.Ctx, cs, preemptorPod, "100m", "100")
		if err != nil {
			t.Fatalf("Failed to update preemptor to deferred resize: %v", err)
		}

		// Preemption should evict victim-non-pdb (200m freed) without violating PDB
		err = wait.PollUntilContextTimeout(testCtx.Ctx, 50*time.Millisecond, 10*time.Second, false,
			podIsGettingEvicted(cs, ns, "victim-non-pdb"))
		if err != nil {
			t.Fatalf("Expected victim-non-pdb to be evicted, got: %v", err)
		}

		// Ensure victim-pdb was NOT evicted
		liveVictimPDB, err := cs.CoreV1().Pods(ns).Get(testCtx.Ctx, "victim-pdb", metav1.GetOptions{})
		if err != nil {
			t.Fatalf("Failed to get victim-pdb: %v", err)
		}
		if liveVictimPDB.DeletionTimestamp != nil {
			t.Errorf("victim-pdb should NOT have been evicted")
		}

		testutils.CleanupPods(testCtx.Ctx, cs, t, []*v1.Pod{victimPDB, victimNonPDB, preemptorPod})
	})

	t.Run("preempt lower priority pod even if it violates PDB when no non-violating alternative exists", func(t *testing.T) {
		testCtx := testutils.InitTestSchedulerWithOptions(t,
			testutils.InitTestAPIServer(t, "def-resize-pdb-2", nil),
			0,
			scheduler.WithProfiles(cfg.Profiles...))
		testutils.SyncSchedulerInformerFactory(testCtx)
		go testCtx.Scheduler.Run(testCtx.SchedulerCtx)
		defer testCtx.SchedulerCloseFn()

		testutils.InitDisruptionController(t, testCtx)
		cs := testCtx.ClientSet
		ns := testCtx.NS.Name

		nodeName := "pdb-node-2"
		node := st.MakeNode().Name(nodeName).Capacity(map[v1.ResourceName]string{
			v1.ResourcePods:   "32",
			v1.ResourceCPU:    "300m",
			v1.ResourceMemory: "300Mi",
		}).Label("node", nodeName).Obj()
		if _, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, node, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create node: %v", err)
		}
		defer func() {
			_ = cs.CoreV1().Nodes().Delete(testCtx.Ctx, nodeName, metav1.DeleteOptions{})
		}()

		// victim-pdb: only available victim on node, requests 200m CPU, protected by PDB minAvailable=1
		victimPDB := initPausePod(&testutils.PausePodConfig{
			Name:      "victim-pdb",
			Namespace: ns,
			Priority:  &asyncframework.LowPriority,
			Labels:    map[string]string{"app": "pdb-app-only"},
			Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
				v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
				v1.ResourceMemory: *resource.NewQuantity(100, resource.BinarySI),
			}},
		})
		victimPDB.Spec.NodeName = nodeName
		victimPDB, err := runPausePod(cs, victimPDB)
		if err != nil {
			t.Fatalf("Failed to run victim-pdb: %v", err)
		}
		addPodReadyCondition(victimPDB)
		if _, err := cs.CoreV1().Pods(ns).UpdateStatus(testCtx.Ctx, victimPDB, metav1.UpdateOptions{}); err != nil {
			t.Fatalf("Failed to update status for victim-pdb: %v", err)
		}

		if err := testutils.WaitCachedPodsStable(testCtx, []*v1.Pod{victimPDB}); err != nil {
			t.Fatalf("Pods not stable in cache: %v", err)
		}

		minAvailable := intstr.FromInt32(1)
		pdb := &policyv1.PodDisruptionBudget{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "pdb-protect-only",
				Namespace: ns,
			},
			Spec: policyv1.PodDisruptionBudgetSpec{
				MinAvailable: &minAvailable,
				Selector:     &metav1.LabelSelector{MatchLabels: map[string]string{"app": "pdb-app-only"}},
			},
		}
		if _, err := cs.PolicyV1().PodDisruptionBudgets(ns).Create(testCtx.Ctx, pdb, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create PDB: %v", err)
		}

		if err := testutils.WaitForPDBsStable(testCtx, []*policyv1.PodDisruptionBudget{pdb}, []int32{1}); err != nil {
			t.Fatalf("PDB not stable: %v", err)
		}

		// Preemptor pod requesting resize from 100m to 300m CPU (needs 200m additional CPU)
		preemptorPod := initPausePod(&testutils.PausePodConfig{
			Name:      "preemptor-pod",
			Namespace: ns,
			Priority:  &asyncframework.HighPriority,
			Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
				v1.ResourceCPU:    *resource.NewMilliQuantity(300, resource.DecimalSI),
				v1.ResourceMemory: *resource.NewQuantity(100, resource.BinarySI),
			}},
		})
		preemptorPod.Spec.NodeName = nodeName
		preemptorPod, err = cs.CoreV1().Pods(ns).Create(testCtx.Ctx, preemptorPod, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create preemptor pod: %v", err)
		}
		preemptorPod, err = updatePodToDeferredResize(testCtx.Ctx, cs, preemptorPod, "100m", "100")
		if err != nil {
			t.Fatalf("Failed to update preemptor to deferred resize: %v", err)
		}

		// Since no non-violating victim exists, preemption proceeds and evicts victim-pdb
		err = wait.PollUntilContextTimeout(testCtx.Ctx, 50*time.Millisecond, 10*time.Second, false,
			podIsGettingEvicted(cs, ns, "victim-pdb"))
		if err != nil {
			t.Fatalf("Expected victim-pdb to be evicted when no non-violating victim exists: %v", err)
		}

		testutils.CleanupPods(testCtx.Ctx, cs, t, []*v1.Pod{victimPDB, preemptorPod})
	})

	t.Run("preemption allowed when PDB allows disruptions", func(t *testing.T) {
		testCtx := testutils.InitTestSchedulerWithOptions(t,
			testutils.InitTestAPIServer(t, "def-resize-pdb-3", nil),
			0,
			scheduler.WithProfiles(cfg.Profiles...))
		testutils.SyncSchedulerInformerFactory(testCtx)
		go testCtx.Scheduler.Run(testCtx.SchedulerCtx)
		defer testCtx.SchedulerCloseFn()

		testutils.InitDisruptionController(t, testCtx)
		cs := testCtx.ClientSet
		ns := testCtx.NS.Name

		nodeName := "pdb-node-3"
		node := st.MakeNode().Name(nodeName).Capacity(map[v1.ResourceName]string{
			v1.ResourcePods:   "32",
			v1.ResourceCPU:    "400m",
			v1.ResourceMemory: "400Mi",
		}).Label("node", nodeName).Obj()
		if _, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, node, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create node: %v", err)
		}
		defer func() {
			_ = cs.CoreV1().Nodes().Delete(testCtx.Ctx, nodeName, metav1.DeleteOptions{})
		}()

		// Two low priority pods matching the same PDB
		v1Pod := initPausePod(&testutils.PausePodConfig{
			Name:      "victim-1",
			Namespace: ns,
			Priority:  &asyncframework.LowPriority,
			Labels:    map[string]string{"app": "pdb-group"},
			Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
				v1.ResourceCPU:    *resource.NewMilliQuantity(100, resource.DecimalSI),
				v1.ResourceMemory: *resource.NewQuantity(100, resource.BinarySI),
			}},
		})
		v1Pod.Spec.NodeName = nodeName
		v1Pod, err := runPausePod(cs, v1Pod)
		if err != nil {
			t.Fatalf("Failed to run victim-1: %v", err)
		}
		addPodReadyCondition(v1Pod)
		if _, err := cs.CoreV1().Pods(ns).UpdateStatus(testCtx.Ctx, v1Pod, metav1.UpdateOptions{}); err != nil {
			t.Fatalf("Failed to update status for victim-1: %v", err)
		}

		v2Pod := initPausePod(&testutils.PausePodConfig{
			Name:      "victim-2",
			Namespace: ns,
			Priority:  &asyncframework.LowPriority,
			Labels:    map[string]string{"app": "pdb-group"},
			Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
				v1.ResourceCPU:    *resource.NewMilliQuantity(100, resource.DecimalSI),
				v1.ResourceMemory: *resource.NewQuantity(100, resource.BinarySI),
			}},
		})
		v2Pod.Spec.NodeName = nodeName
		v2Pod, err = runPausePod(cs, v2Pod)
		if err != nil {
			t.Fatalf("Failed to run victim-2: %v", err)
		}
		addPodReadyCondition(v2Pod)
		if _, err := cs.CoreV1().Pods(ns).UpdateStatus(testCtx.Ctx, v2Pod, metav1.UpdateOptions{}); err != nil {
			t.Fatalf("Failed to update status for victim-2: %v", err)
		}

		if err := testutils.WaitCachedPodsStable(testCtx, []*v1.Pod{v1Pod, v2Pod}); err != nil {
			t.Fatalf("Pods not stable in cache: %v", err)
		}

		// minAvailable=1 with 2 pods ready -> disruptionsAllowed = 1
		minAvailable := intstr.FromInt32(1)
		pdb := &policyv1.PodDisruptionBudget{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "pdb-group",
				Namespace: ns,
			},
			Spec: policyv1.PodDisruptionBudgetSpec{
				MinAvailable: &minAvailable,
				Selector:     &metav1.LabelSelector{MatchLabels: map[string]string{"app": "pdb-group"}},
			},
		}
		if _, err := cs.PolicyV1().PodDisruptionBudgets(ns).Create(testCtx.Ctx, pdb, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create PDB: %v", err)
		}

		if err := testutils.WaitForPDBsStable(testCtx, []*policyv1.PodDisruptionBudget{pdb}, []int32{2}); err != nil {
			t.Fatalf("PDB not stable: %v", err)
		}

		// Preemptor pod requesting resize from 200m to 300m CPU (needs 100m additional CPU)
		preemptorPod := initPausePod(&testutils.PausePodConfig{
			Name:      "preemptor-pod",
			Namespace: ns,
			Priority:  &asyncframework.HighPriority,
			Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
				v1.ResourceCPU:    *resource.NewMilliQuantity(300, resource.DecimalSI),
				v1.ResourceMemory: *resource.NewQuantity(100, resource.BinarySI),
			}},
		})
		preemptorPod.Spec.NodeName = nodeName
		preemptorPod, err = cs.CoreV1().Pods(ns).Create(testCtx.Ctx, preemptorPod, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create preemptor pod: %v", err)
		}
		preemptorPod, err = updatePodToDeferredResize(testCtx.Ctx, cs, preemptorPod, "200m", "100")
		if err != nil {
			t.Fatalf("Failed to update preemptor to deferred resize: %v", err)
		}

		// Preemption should evict exactly 1 victim
		var evictedCount int
		err = wait.PollUntilContextTimeout(testCtx.Ctx, 50*time.Millisecond, 10*time.Second, false, func(ctx context.Context) (bool, error) {
			evictedCount = 0
			for _, name := range []string{"victim-1", "victim-2"} {
				p, err := cs.CoreV1().Pods(ns).Get(ctx, name, metav1.GetOptions{})
				if err == nil && p.DeletionTimestamp != nil {
					evictedCount++
				}
			}
			return evictedCount >= 1, nil
		})
		if err != nil {
			t.Fatalf("Expected at least 1 victim to be evicted, got %d evictions: %v", evictedCount, err)
		}

		// Wait briefly and verify only 1 was evicted (no over-eviction violating PDB)
		time.Sleep(300 * time.Millisecond)
		evictedCount = 0
		for _, name := range []string{"victim-1", "victim-2"} {
			p, err := cs.CoreV1().Pods(ns).Get(testCtx.Ctx, name, metav1.GetOptions{})
			if err == nil && p.DeletionTimestamp != nil {
				evictedCount++
			}
		}
		if evictedCount != 1 {
			t.Errorf("Expected exactly 1 pod evicted to respect PDB disruption budget, but got %d", evictedCount)
		}

		testutils.CleanupPods(testCtx.Ctx, cs, t, []*v1.Pod{v1Pod, v2Pod, preemptorPod})
	})
}

func TestDeferredResizePodPreemption_ConcurrentDeferredResizes(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.InPlacePodVerticalScaling:                    true,
		features.InPlacePodVerticalScalingSchedulerPreemption: true,
	})

	cfg := configtesting.V1ToInternalWithDefaults(t, configv1.KubeSchedulerConfiguration{
		Profiles: []configv1.KubeSchedulerProfile{{
			SchedulerName: new(v1.DefaultSchedulerName),
		}},
	})

	t.Run("higher priority resizing pod preempts before lower priority resizing pod", func(t *testing.T) {
		testCtx := testutils.InitTestSchedulerWithOptions(t,
			testutils.InitTestAPIServer(t, "def-resize-concurrent-1", nil),
			0,
			scheduler.WithProfiles(cfg.Profiles...))
		testutils.SyncSchedulerInformerFactory(testCtx)
		go testCtx.Scheduler.Run(testCtx.SchedulerCtx)
		defer testCtx.SchedulerCloseFn()

		cs := testCtx.ClientSet
		ns := testCtx.NS.Name

		nodeName := "concurrent-node-1"
		node := st.MakeNode().Name(nodeName).Capacity(map[v1.ResourceName]string{
			v1.ResourcePods:   "32",
			v1.ResourceCPU:    "500m",
			v1.ResourceMemory: "500Mi",
		}).Label("node", nodeName).Obj()
		if _, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, node, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create node: %v", err)
		}
		defer func() {
			_ = cs.CoreV1().Nodes().Delete(testCtx.Ctx, nodeName, metav1.DeleteOptions{})
		}()

		// Victim pod requesting 200m CPU (low priority: 0)
		victim := initPausePod(&testutils.PausePodConfig{
			Name:      "low-victim",
			Namespace: ns,
			Priority:  &asyncframework.LowPriority,
			Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
				v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
				v1.ResourceMemory: *resource.NewQuantity(100, resource.BinarySI),
			}},
		})
		victim.Spec.NodeName = nodeName
		victim, err := runPausePod(cs, victim)
		if err != nil {
			t.Fatalf("Failed to run low-victim: %v", err)
		}

		// Preemptor 1: High priority (300), allocated 100m, requests 300m (needs 200m additional CPU)
		highPreemptor := initPausePod(&testutils.PausePodConfig{
			Name:      "high-preemptor",
			Namespace: ns,
			Priority:  &asyncframework.HighPriority,
			Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
				v1.ResourceCPU:    *resource.NewMilliQuantity(300, resource.DecimalSI),
				v1.ResourceMemory: *resource.NewQuantity(100, resource.BinarySI),
			}},
		})
		highPreemptor.Spec.NodeName = nodeName
		highPreemptor, err = cs.CoreV1().Pods(ns).Create(testCtx.Ctx, highPreemptor, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create high-preemptor: %v", err)
		}

		// Preemptor 2: Mid priority (200), allocated 100m, requests 200m (needs 100m additional CPU)
		midPreemptor := initPausePod(&testutils.PausePodConfig{
			Name:      "mid-preemptor",
			Namespace: ns,
			Priority:  &asyncframework.MediumPriority,
			Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
				v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
				v1.ResourceMemory: *resource.NewQuantity(100, resource.BinarySI),
			}},
		})
		midPreemptor.Spec.NodeName = nodeName
		midPreemptor, err = cs.CoreV1().Pods(ns).Create(testCtx.Ctx, midPreemptor, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create mid-preemptor: %v", err)
		}

		// Transition both preemptors to deferred resize state concurrently
		highPreemptor, err = updatePodToDeferredResize(testCtx.Ctx, cs, highPreemptor, "100m", "100")
		if err != nil {
			t.Fatalf("Failed to update high-preemptor status: %v", err)
		}
		midPreemptor, err = updatePodToDeferredResize(testCtx.Ctx, cs, midPreemptor, "100m", "100")
		if err != nil {
			t.Fatalf("Failed to update mid-preemptor status: %v", err)
		}

		// Low-victim should be evicted by the scheduler
		err = wait.PollUntilContextTimeout(testCtx.Ctx, 50*time.Millisecond, 10*time.Second, false,
			podIsGettingEvicted(cs, ns, "low-victim"))
		if err != nil {
			t.Fatalf("Expected low-victim to be evicted: %v", err)
		}

		// High preemptor is prioritized; mid-preemptor cannot preempt high-preemptor and remains deferred
		time.Sleep(300 * time.Millisecond)
		liveHigh, err := cs.CoreV1().Pods(ns).Get(testCtx.Ctx, "high-preemptor", metav1.GetOptions{})
		if err != nil {
			t.Fatalf("Failed to get high-preemptor: %v", err)
		}
		if liveHigh.DeletionTimestamp != nil {
			t.Errorf("high-preemptor should NOT be evicted")
		}

		liveMid, err := cs.CoreV1().Pods(ns).Get(testCtx.Ctx, "mid-preemptor", metav1.GetOptions{})
		if err != nil {
			t.Fatalf("Failed to get mid-preemptor: %v", err)
		}
		if liveMid.DeletionTimestamp != nil {
			t.Errorf("mid-preemptor should NOT be evicted")
		}

		// mid-preemptor should remain queued in the scheduler
		_, found := testCtx.Scheduler.SchedulingQueue.GetPod(testCtx.Ctx, midPreemptor.Name, midPreemptor.Namespace, nil)
		if !found {
			t.Errorf("Expected mid-preemptor pod to remain in scheduling queue")
		}

		testutils.CleanupPods(testCtx.Ctx, cs, t, []*v1.Pod{victim, highPreemptor, midPreemptor})
	})

	t.Run("concurrent resizing pods preempt distinct victims without over-eviction", func(t *testing.T) {
		testCtx := testutils.InitTestSchedulerWithOptions(t,
			testutils.InitTestAPIServer(t, "def-resize-concurrent-2", nil),
			0,
			scheduler.WithProfiles(cfg.Profiles...))
		testutils.SyncSchedulerInformerFactory(testCtx)
		go testCtx.Scheduler.Run(testCtx.SchedulerCtx)
		defer testCtx.SchedulerCloseFn()

		cs := testCtx.ClientSet
		ns := testCtx.NS.Name

		nodeName := "concurrent-node-2"
		// Node capacity: 500m CPU
		node := st.MakeNode().Name(nodeName).Capacity(map[v1.ResourceName]string{
			v1.ResourcePods:   "32",
			v1.ResourceCPU:    "500m",
			v1.ResourceMemory: "500Mi",
		}).Label("node", nodeName).Obj()
		if _, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, node, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create node: %v", err)
		}
		defer func() {
			_ = cs.CoreV1().Nodes().Delete(testCtx.Ctx, nodeName, metav1.DeleteOptions{})
		}()

		// 3 victim pods (100m CPU each, priority 0)
		var victims []*v1.Pod
		for i := 1; i <= 3; i++ {
			v := initPausePod(&testutils.PausePodConfig{
				Name:      fmt.Sprintf("victim-%d", i),
				Namespace: ns,
				Priority:  &asyncframework.LowPriority,
				Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
					v1.ResourceCPU:    *resource.NewMilliQuantity(100, resource.DecimalSI),
					v1.ResourceMemory: *resource.NewQuantity(100, resource.BinarySI),
				}},
			})
			v.Spec.NodeName = nodeName
			v, err := runPausePod(cs, v)
			if err != nil {
				t.Fatalf("Failed to run victim-%d: %v", i, err)
			}
			victims = append(victims, v)
		}

		// 2 resizing pods (allocated 100m CPU each, requesting 200m CPU each -> each needs 100m CPU)
		// Total extra CPU needed = 200m CPU -> exactly 2 victims should be evicted, leaving 1 untouched.
		preemptor1 := initPausePod(&testutils.PausePodConfig{
			Name:      "preemptor-1",
			Namespace: ns,
			Priority:  &asyncframework.HighPriority,
			Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
				v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
				v1.ResourceMemory: *resource.NewQuantity(100, resource.BinarySI),
			}},
		})
		preemptor1.Spec.NodeName = nodeName
		preemptor1, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, preemptor1, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create preemptor-1: %v", err)
		}

		preemptor2 := initPausePod(&testutils.PausePodConfig{
			Name:      "preemptor-2",
			Namespace: ns,
			Priority:  &asyncframework.HighPriority,
			Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
				v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
				v1.ResourceMemory: *resource.NewQuantity(100, resource.BinarySI),
			}},
		})
		preemptor2.Spec.NodeName = nodeName
		preemptor2, err = cs.CoreV1().Pods(ns).Create(testCtx.Ctx, preemptor2, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create preemptor-2: %v", err)
		}

		preemptor1, err = updatePodToDeferredResize(testCtx.Ctx, cs, preemptor1, "100m", "100")
		if err != nil {
			t.Fatalf("Failed to update preemptor-1 status: %v", err)
		}
		preemptor2, err = updatePodToDeferredResize(testCtx.Ctx, cs, preemptor2, "100m", "100")
		if err != nil {
			t.Fatalf("Failed to update preemptor-2 status: %v", err)
		}

		// Wait for 2 victims to be evicted
		err = wait.PollUntilContextTimeout(testCtx.Ctx, 50*time.Millisecond, 10*time.Second, false, func(ctx context.Context) (bool, error) {
			evictedCount := 0
			for _, v := range victims {
				p, err := cs.CoreV1().Pods(ns).Get(ctx, v.Name, metav1.GetOptions{})
				if err == nil && p.DeletionTimestamp != nil {
					evictedCount++
				}
			}
			return evictedCount >= 2, nil
		})
		if err != nil {
			t.Fatalf("Expected at least 2 victims to be evicted: %v", err)
		}

		// Verify that exactly 2 victims are evicted and not all 3 (no over-eviction)
		time.Sleep(300 * time.Millisecond)
		evictedCount := 0
		for _, v := range victims {
			p, err := cs.CoreV1().Pods(ns).Get(testCtx.Ctx, v.Name, metav1.GetOptions{})
			if err == nil && p.DeletionTimestamp != nil {
				evictedCount++
			}
		}
		if evictedCount != 2 {
			t.Errorf("Expected exactly 2 victims evicted, but got %d", evictedCount)
		}

		testutils.CleanupPods(testCtx.Ctx, cs, t, append(victims, preemptor1, preemptor2))
	})
}

func TestDeferredResizePodPreemption_PodGroupVictims(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.InPlacePodVerticalScaling:                    true,
		features.InPlacePodVerticalScalingSchedulerPreemption: true,
		features.GenericWorkload:                              true,
	})

	cfg := configtesting.V1ToInternalWithDefaults(t, configv1.KubeSchedulerConfiguration{
		Profiles: []configv1.KubeSchedulerProfile{{
			SchedulerName: new(v1.DefaultSchedulerName),
		}},
	})

	t.Run("disruption mode all evicts all podgroup members across nodes", func(t *testing.T) {
		testCtx := testutils.InitTestSchedulerWithOptions(t,
			testutils.InitTestAPIServer(t, "def-resize-podgroup-1", nil),
			0,
			scheduler.WithProfiles(cfg.Profiles...))
		testutils.SyncSchedulerInformerFactory(testCtx)
		go testCtx.Scheduler.Run(testCtx.SchedulerCtx)
		defer testCtx.SchedulerCloseFn()

		cs := testCtx.ClientSet
		ns := testCtx.NS.Name

		// Create two nodes
		for _, nodeName := range []string{"pg-node1-1", "pg-node2-1"} {
			node := st.MakeNode().Name(nodeName).Capacity(map[v1.ResourceName]string{
				v1.ResourcePods:   "32",
				v1.ResourceCPU:    "400m",
				v1.ResourceMemory: "400Mi",
			}).Label("node", nodeName).Obj()
			if _, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, node, metav1.CreateOptions{}); err != nil {
				t.Fatalf("Failed to create node %s: %v", nodeName, err)
			}
			nodeName := nodeName
			defer func() {
				_ = cs.CoreV1().Nodes().Delete(testCtx.Ctx, nodeName, metav1.DeleteOptions{})
			}()
		}

		// Create PodGroup with DisruptionModeAll
		pg := st.MakePodGroup().Name("pg-all").Namespace(ns).Priority(asyncframework.LowPriority).BasicPolicy().DisruptionModeAll().Obj()
		if _, err := cs.SchedulingV1beta1().PodGroups(ns).Create(testCtx.Ctx, pg, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create PodGroup pg-all: %v", err)
		}

		// Pod 1 on pg-node1-1
		pgMember1 := initPausePod(&testutils.PausePodConfig{
			Name:         "pg-member-1",
			Namespace:    ns,
			Priority:     &asyncframework.LowPriority,
			PodGroupName: "pg-all",
			Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
				v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
				v1.ResourceMemory: *resource.NewQuantity(100, resource.BinarySI),
			}},
		})
		pgMember1.Spec.NodeName = "pg-node1-1"
		pgMember1, err := runPausePod(cs, pgMember1)
		if err != nil {
			t.Fatalf("Failed to run pg-member-1: %v", err)
		}

		// Pod 2 on pg-node2-1
		pgMember2 := initPausePod(&testutils.PausePodConfig{
			Name:         "pg-member-2",
			Namespace:    ns,
			Priority:     &asyncframework.LowPriority,
			PodGroupName: "pg-all",
			Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
				v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
				v1.ResourceMemory: *resource.NewQuantity(100, resource.BinarySI),
			}},
		})
		pgMember2.Spec.NodeName = "pg-node2-1"
		pgMember2, err = runPausePod(cs, pgMember2)
		if err != nil {
			t.Fatalf("Failed to run pg-member-2: %v", err)
		}

		// Preemptor pod on pg-node1-1: requests resize from 100m to 300m CPU (needs 200m additional CPU on pg-node1-1)
		preemptorPod := initPausePod(&testutils.PausePodConfig{
			Name:      "preemptor-pod",
			Namespace: ns,
			Priority:  &asyncframework.HighPriority,
			Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
				v1.ResourceCPU:    *resource.NewMilliQuantity(300, resource.DecimalSI),
				v1.ResourceMemory: *resource.NewQuantity(100, resource.BinarySI),
			}},
		})
		preemptorPod.Spec.NodeName = "pg-node1-1"
		preemptorPod, err = cs.CoreV1().Pods(ns).Create(testCtx.Ctx, preemptorPod, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create preemptor-pod: %v", err)
		}
		preemptorPod, err = updatePodToDeferredResize(testCtx.Ctx, cs, preemptorPod, "100m", "100")
		if err != nil {
			t.Fatalf("Failed to update preemptor-pod status: %v", err)
		}

		// Preemption on pg-node1-1 should evict BOTH pg-member-1 (on node 1) AND pg-member-2 (on node 2)
		for _, victimName := range []string{"pg-member-1", "pg-member-2"} {
			err = wait.PollUntilContextTimeout(testCtx.Ctx, 50*time.Millisecond, 10*time.Second, false,
				podIsGettingEvicted(cs, ns, victimName))
			if err != nil {
				t.Fatalf("Expected pod %s in DisruptionModeAll PodGroup to be evicted, got: %v", victimName, err)
			}
		}

		testutils.CleanupPods(testCtx.Ctx, cs, t, []*v1.Pod{pgMember1, pgMember2, preemptorPod})
	})

	t.Run("disruption mode single evicts only target node podgroup member", func(t *testing.T) {
		testCtx := testutils.InitTestSchedulerWithOptions(t,
			testutils.InitTestAPIServer(t, "def-resize-podgroup-2", nil),
			0,
			scheduler.WithProfiles(cfg.Profiles...))
		testutils.SyncSchedulerInformerFactory(testCtx)
		go testCtx.Scheduler.Run(testCtx.SchedulerCtx)
		defer testCtx.SchedulerCloseFn()

		cs := testCtx.ClientSet
		ns := testCtx.NS.Name

		// Create two nodes
		for _, nodeName := range []string{"pg-node1-2", "pg-node2-2"} {
			node := st.MakeNode().Name(nodeName).Capacity(map[v1.ResourceName]string{
				v1.ResourcePods:   "32",
				v1.ResourceCPU:    "400m",
				v1.ResourceMemory: "400Mi",
			}).Label("node", nodeName).Obj()
			if _, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, node, metav1.CreateOptions{}); err != nil {
				t.Fatalf("Failed to create node %s: %v", nodeName, err)
			}
			nodeName := nodeName
			defer func() {
				_ = cs.CoreV1().Nodes().Delete(testCtx.Ctx, nodeName, metav1.DeleteOptions{})
			}()
		}

		// Create PodGroup with DisruptionModeSingle
		pg := st.MakePodGroup().Name("pg-single").Namespace(ns).Priority(asyncframework.LowPriority).BasicPolicy().DisruptionModeSingle().Obj()
		if _, err := cs.SchedulingV1beta1().PodGroups(ns).Create(testCtx.Ctx, pg, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create PodGroup pg-single: %v", err)
		}

		// Pod 1 on pg-node1-2
		pgMember1 := initPausePod(&testutils.PausePodConfig{
			Name:         "pg-member-1",
			Namespace:    ns,
			Priority:     &asyncframework.LowPriority,
			PodGroupName: "pg-single",
			Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
				v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
				v1.ResourceMemory: *resource.NewQuantity(100, resource.BinarySI),
			}},
		})
		pgMember1.Spec.NodeName = "pg-node1-2"
		pgMember1, err := runPausePod(cs, pgMember1)
		if err != nil {
			t.Fatalf("Failed to run pg-member-1: %v", err)
		}

		// Pod 2 on pg-node2-2
		pgMember2 := initPausePod(&testutils.PausePodConfig{
			Name:         "pg-member-2",
			Namespace:    ns,
			Priority:     &asyncframework.LowPriority,
			PodGroupName: "pg-single",
			Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
				v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
				v1.ResourceMemory: *resource.NewQuantity(100, resource.BinarySI),
			}},
		})
		pgMember2.Spec.NodeName = "pg-node2-2"
		pgMember2, err = runPausePod(cs, pgMember2)
		if err != nil {
			t.Fatalf("Failed to run pg-member-2: %v", err)
		}

		// Preemptor pod on pg-node1-2: requests resize from 100m to 300m CPU (needs 200m additional CPU on pg-node1-2)
		preemptorPod := initPausePod(&testutils.PausePodConfig{
			Name:      "preemptor-pod",
			Namespace: ns,
			Priority:  &asyncframework.HighPriority,
			Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
				v1.ResourceCPU:    *resource.NewMilliQuantity(300, resource.DecimalSI),
				v1.ResourceMemory: *resource.NewQuantity(100, resource.BinarySI),
			}},
		})
		preemptorPod.Spec.NodeName = "pg-node1-2"
		preemptorPod, err = cs.CoreV1().Pods(ns).Create(testCtx.Ctx, preemptorPod, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create preemptor-pod: %v", err)
		}
		preemptorPod, err = updatePodToDeferredResize(testCtx.Ctx, cs, preemptorPod, "100m", "100")
		if err != nil {
			t.Fatalf("Failed to update preemptor-pod status: %v", err)
		}

		// Preemption on pg-node1-2 should evict pg-member-1 (on node 1)
		err = wait.PollUntilContextTimeout(testCtx.Ctx, 50*time.Millisecond, 10*time.Second, false,
			podIsGettingEvicted(cs, ns, "pg-member-1"))
		if err != nil {
			t.Fatalf("Expected pg-member-1 on target node to be evicted: %v", err)
		}

		// pg-member-2 on pg-node2-2 must NOT be evicted because of DisruptionModeSingle
		time.Sleep(300 * time.Millisecond)
		liveMember2, err := cs.CoreV1().Pods(ns).Get(testCtx.Ctx, "pg-member-2", metav1.GetOptions{})
		if err != nil {
			t.Fatalf("Failed to get pg-member-2: %v", err)
		}
		if liveMember2.DeletionTimestamp != nil {
			t.Errorf("pg-member-2 on other node should NOT be evicted in DisruptionModeSingle")
		}

		testutils.CleanupPods(testCtx.Ctx, cs, t, []*v1.Pod{pgMember1, pgMember2, preemptorPod})
	})
}
