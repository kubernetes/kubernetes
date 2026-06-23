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
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	configv1 "k8s.io/kube-scheduler/config/v1"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler"
	configtesting "k8s.io/kubernetes/pkg/scheduler/apis/config/testing"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
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
			SchedulerName: ptr.To(v1.DefaultSchedulerName),
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
					Priority: &lowPriority,
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(100, resource.DecimalSI)},
					},
				}),
			},
			preemptorConfig: &testutils.PausePodConfig{
				Name:     "preemptor-pod",
				Priority: &highPriority,
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
					Priority: &lowPriority,
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(100, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(50, resource.DecimalSI)},
					},
				}),
				initPausePod(&testutils.PausePodConfig{
					Name:     "victim-2",
					Priority: &lowPriority,
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(100, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(50, resource.DecimalSI)},
					},
				}),
			},
			preemptorConfig: &testutils.PausePodConfig{
				Name:     "preemptor-pod",
				Priority: &highPriority,
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
					Priority: &lowPriority,
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(100, resource.DecimalSI)},
					},
				}),
			},
			preemptorConfig: &testutils.PausePodConfig{
				Name:             "preemptor-pod",
				Priority:         &highPriority,
				PreemptionPolicy: ptr.To(v1.PreemptNever),
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
				Priority: &highPriority,
				Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
					v1.ResourceCPU:    *resource.NewMilliQuantity(300, resource.DecimalSI),
					v1.ResourceMemory: *resource.NewQuantity(100, resource.DecimalSI)},
				},
			},
			expectEvictedNames: nil,
		},
		{
			name:            "version skew node-level fallback disables preemption",
			nodeCapacityCPU: "300m",
			nodeCapacityMem: "300",
			nodePreemptionPolicy: &v1.NodePodPreemptionPolicy{
				DisableResizePreemption: []string{"test-skew-operator"},
			},
			existingPods: []*v1.Pod{
				initPausePod(&testutils.PausePodConfig{
					Name:     "victim-1",
					Priority: &lowPriority,
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(100, resource.DecimalSI)},
					},
				}),
			},
			preemptorConfig: &testutils.PausePodConfig{
				Name:     "preemptor-pod",
				Priority: &highPriority,
				Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
					v1.ResourceCPU:    *resource.NewMilliQuantity(300, resource.DecimalSI),
					v1.ResourceMemory: *resource.NewQuantity(100, resource.DecimalSI)},
				},
			},
			expectEvictedNames: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Initialize API server and Scheduler
			testCtx := testutils.InitTestSchedulerWithOptions(t,
				testutils.InitTestAPIServer(t, "deferred-preemption-test", nil),
				0,
				scheduler.WithProfiles(cfg.Profiles...),
			)
			defer testCtx.SchedulerCloseFn()
			testutils.SyncSchedulerInformerFactory(testCtx)
			go testCtx.Scheduler.Run(testCtx.Ctx)

			cs := testCtx.ClientSet

			// Create node
			nodeRes := map[v1.ResourceName]string{
				v1.ResourcePods:   "32",
				v1.ResourceCPU:    tt.nodeCapacityCPU,
				v1.ResourceMemory: tt.nodeCapacityMem,
			}
			nodeObject := st.MakeNode().Name("node1").Capacity(nodeRes).Label("node", "node1").Obj()
			if tt.nodePreemptionPolicy != nil {
				nodeObject.Spec.PodPreemptionPolicy = tt.nodePreemptionPolicy
			}
			if _, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, nodeObject, metav1.CreateOptions{}); err != nil {
				t.Fatalf("Failed to create node: %v", err)
			}

			// Create and run existing pods (if any) on node1
			var pods []*v1.Pod
			for _, p := range tt.existingPods {
				p.Namespace = testCtx.NS.Name
				p.Spec.NodeName = "node1"
				runningPod, err := runPausePod(cs, p)
				if err != nil {
					t.Fatalf("Failed running pause pod %v: %v", p.Name, err)
				}
				pods = append(pods, runningPod)
			}

			// Create preemptor/resizing pod already scheduled to node1
			tt.preemptorConfig.NodeName = "node1"
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
					err = wait.PollUntilContextTimeout(testCtx.Ctx, 200*time.Millisecond, wait.ForeverTestTimeout, false,
						podIsGettingEvicted(cs, testCtx.NS.Name, name))
					if err != nil {
						t.Errorf("Expected pod %q to be evicted/deleted, but it was not", name)
					}
				}
			} else {
				// Wait to confirm NO evictions happen
				time.Sleep(2 * time.Second)
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

			// Verify the pod remains parked in the scheduler queue (or not enqueued if preemption is disabled)
			queue := testCtx.Scheduler.SchedulingQueue
			_, found := queue.GetPod(preemptorPod.Name, preemptorPod.Namespace, nil)
			expectFound := tt.name != "version skew node-level fallback disables preemption"
			if found != expectFound {
				t.Errorf("Expected preemptor pod found in scheduling queue to be %t, but got %t", expectFound, found)
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

func setUpPreemptionTest(t *testing.T, nodeCPUCapacity, otherPodCPURequest, deferredPodCPURequest, deferredPodCPUAllocated string) (*testutils.TestContext, *v1.Pod, *v1.Pod, *v1.Pod) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.InPlacePodVerticalScaling:                    true,
		features.InPlacePodVerticalScalingSchedulerPreemption: true,
	})

	cfg := configtesting.V1ToInternalWithDefaults(t, configv1.KubeSchedulerConfiguration{
		Profiles: []configv1.KubeSchedulerProfile{{
			SchedulerName: ptr.To(v1.DefaultSchedulerName),
		}},
	})

	testCtx := testutils.InitTestSchedulerWithOptions(t,
		testutils.InitTestAPIServer(t, "deferred-queueing-test", nil),
		0,
		scheduler.WithProfiles(cfg.Profiles...),
	)
	testutils.SyncSchedulerInformerFactory(testCtx)
	go testCtx.Scheduler.Run(testCtx.Ctx)

	cs := testCtx.ClientSet

	// Create node1
	nodeObject1 := st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{
		v1.ResourcePods: "32",
		v1.ResourceCPU:  nodeCPUCapacity,
	}).Obj()
	if _, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, nodeObject1, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create node1: %v", err)
	}

	// Create node2 (irrelevant node)
	nodeObject2 := st.MakeNode().Name("node2").Capacity(map[v1.ResourceName]string{
		v1.ResourcePods: "32",
		v1.ResourceCPU:  "500m",
	}).Obj()
	if _, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, nodeObject2, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create node2: %v", err)
	}

	// Create 'other-pod' utilizing node1
	other := initPausePod(&testutils.PausePodConfig{
		Name:     "other-pod",
		NodeName: "node1",
		Priority: &highPriority,
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
		Name:     "irrelevant-pod",
		NodeName: "node2",
		Priority: &highPriority,
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
		Name:     "deferred-pod",
		NodeName: "node1",
		Priority: &lowPriority,
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
	err = wait.PollUntilContextTimeout(testCtx.Ctx, 200*time.Millisecond, wait.ForeverTestTimeout, false, func(context.Context) (bool, error) {
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

	return testCtx, pod, other, irrelevant
}

func TestDeferredResizeQueueingHints(t *testing.T) {
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
			name:               "irrelevant pod scale up ignores queue",
			nodeCPUCapacity:    "300m",
			deferredPodRequest: "200m",
			expectIncrement:    false,
			trigger: func(ctx context.Context, cs clientset.Interface, ns string, pod, other, irrelevant *v1.Pod) error {
				p, err := cs.CoreV1().Pods(ns).Get(ctx, irrelevant.Name, metav1.GetOptions{})
				if err != nil {
					return err
				}
				p.Spec.Containers[0].Resources.Requests = v1.ResourceList{v1.ResourceCPU: *resource.NewMilliQuantity(300, resource.DecimalSI)}
				_, err = cs.CoreV1().Pods(ns).UpdateResize(ctx, p.Name, p, metav1.UpdateOptions{})
				return err
			},
		},
		{
			name:               "assigned pod scale up on same node wakes queue",
			nodeCPUCapacity:    "300m",
			deferredPodRequest: "200m",
			expectIncrement:    true,
			trigger: func(ctx context.Context, cs clientset.Interface, ns string, pod, other, irrelevant *v1.Pod) error {
				p, err := cs.CoreV1().Pods(ns).Get(ctx, other.Name, metav1.GetOptions{})
				if err != nil {
					return err
				}
				p.Spec.Containers[0].Resources.Requests = v1.ResourceList{v1.ResourceCPU: *resource.NewMilliQuantity(250, resource.DecimalSI)}
				_, err = cs.CoreV1().Pods(ns).UpdateResize(ctx, p.Name, p, metav1.UpdateOptions{})
				return err
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
			name:            "irrelevant node capacity increase ignores queue",
			expectIncrement: false,
			trigger: func(ctx context.Context, cs clientset.Interface, ns string, pod, other, irrelevant *v1.Pod) error {
				n, err := cs.CoreV1().Nodes().Get(ctx, "node2", metav1.GetOptions{})
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
			name:            "assigned node capacity increase wakes queue",
			expectIncrement: true,
			trigger: func(ctx context.Context, cs clientset.Interface, ns string, pod, other, irrelevant *v1.Pod) error {
				n, err := cs.CoreV1().Nodes().Get(ctx, "node1", metav1.GetOptions{})
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
		{
			name:               "target pod spec scale up wakes queue",
			nodeCPUCapacity:    "500m",
			otherPodCPURequest: "200m",
			deferredPodRequest: "400m",
			expectIncrement:    true,
			trigger: func(ctx context.Context, cs clientset.Interface, ns string, pod, other, irrelevant *v1.Pod) error {
				p, err := cs.CoreV1().Pods(ns).Get(ctx, pod.Name, metav1.GetOptions{})
				if err != nil {
					return err
				}
				p.Spec.Containers[0].Resources.Requests = v1.ResourceList{v1.ResourceCPU: *resource.NewMilliQuantity(450, resource.DecimalSI)}
				_, err = cs.CoreV1().Pods(ns).UpdateResize(ctx, p.Name, p, metav1.UpdateOptions{})
				return err
			},
		},
	}

	for _, tc := range tests {
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
			testCtx, pod, other, irrelevant := setUpPreemptionTest(t, nodeCap, otherReq, defReq, "50m")
			defer testCtx.SchedulerCloseFn()

			cs := testCtx.ClientSet
			queue := testCtx.Scheduler.SchedulingQueue

			queuedPod, found := queue.GetPod(pod.Name, pod.Namespace, nil)
			if !found {
				t.Fatalf("Pod not found in queue")
			}
			initialAttempts := queuedPod.Attempts

			if err := tc.trigger(testCtx.Ctx, cs, testCtx.NS.Name, pod, other, irrelevant); err != nil {
				t.Fatalf("Failed triggering event: %v", err)
			}

			if tc.expectIncrement {
				err := wait.PollUntilContextTimeout(testCtx.Ctx, 200*time.Millisecond, wait.ForeverTestTimeout, false, func(context.Context) (bool, error) {
					qPod, found := queue.GetPod(pod.Name, pod.Namespace, nil)
					return found && qPod.Attempts > initialAttempts, nil
				})
				if err != nil {
					t.Fatalf("Expected queue attempts to increment after trigger")
				}
			} else {
				time.Sleep(2 * time.Second)
				qPod, found := queue.GetPod(pod.Name, pod.Namespace, nil)
				if found && qPod.Attempts > initialAttempts {
					t.Fatalf("Expected attempts not to increment, but went from %d to %d", initialAttempts, qPod.Attempts)
				}
			}

			testutils.CleanupPods(testCtx.Ctx, cs, t, []*v1.Pod{pod, other, irrelevant})
		})
	}
}

func TestDeferredResizeQueueingHandlers(t *testing.T) {
	t.Run("AddPod handler enqueues existing deferred pod on scheduler startup", func(t *testing.T) {
		featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
			features.InPlacePodVerticalScaling:                    true,
			features.InPlacePodVerticalScalingSchedulerPreemption: true,
		})

		cfg := configtesting.V1ToInternalWithDefaults(t, configv1.KubeSchedulerConfiguration{
			Profiles: []configv1.KubeSchedulerProfile{{
				SchedulerName: ptr.To(v1.DefaultSchedulerName),
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
			Priority: &lowPriority,
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
			_, found = queue.GetPod(pod.Name, pod.Namespace, nil)
			return found, nil
		})
		if err != nil || !found {
			t.Fatalf("Expected pod to be queued automatically by AddPod handler on scheduler startup, err: %v, found: %t", err, found)
		}

		testutils.CleanupPods(testCtx.Ctx, cs, t, []*v1.Pod{pod})
	})

	t.Run("AddPod handler ignores existing deferred pod on scheduler startup when preemption is disabled", func(t *testing.T) {
		featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
			features.InPlacePodVerticalScaling:                    true,
			features.InPlacePodVerticalScalingSchedulerPreemption: true,
		})

		cfg := configtesting.V1ToInternalWithDefaults(t, configv1.KubeSchedulerConfiguration{
			Profiles: []configv1.KubeSchedulerProfile{{
				SchedulerName: ptr.To(v1.DefaultSchedulerName),
			}},
		})

		testCtx := testutils.InitTestAPIServer(t, "deferred-handlers-test", nil)

		cs := testCtx.ClientSet

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
			Priority: &lowPriority,
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
			{
				Type:   v1.PodResizePreemptionDisabled,
				Status: v1.ConditionTrue,
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

		testCtx = testutils.InitTestSchedulerWithOptions(t, testCtx, 0, scheduler.WithProfiles(cfg.Profiles...))
		defer testCtx.SchedulerCloseFn()
		testutils.SyncSchedulerInformerFactory(testCtx)

		queue := testCtx.Scheduler.SchedulingQueue
		time.Sleep(1 * time.Second)
		if _, found := queue.GetPod(pod.Name, pod.Namespace, nil); found {
			t.Fatalf("Expected pod with PodResizePreemptionDisabled=True not to be enqueued on startup")
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
				SchedulerName: ptr.To(v1.DefaultSchedulerName),
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
			Priority: &lowPriority,
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
		time.Sleep(1 * time.Second)
		if _, found := queue.GetPod(pod.Name, pod.Namespace, nil); found {
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
			_, found = queue.GetPod(pod.Name, pod.Namespace, nil)
			return found, nil
		})
		if err != nil || !found {
			t.Fatalf("Expected pod to be enqueued automatically by UpdatePod handler on deferred status transition")
		}

		testutils.CleanupPods(testCtx.Ctx, cs, t, []*v1.Pod{pod})
	})

	t.Run("UpdatePod handler removes pod from queue when PodResizePreemptionDisabled transitions to True", func(t *testing.T) {
		featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
			features.InPlacePodVerticalScaling:                    true,
			features.InPlacePodVerticalScalingSchedulerPreemption: true,
		})

		cfg := configtesting.V1ToInternalWithDefaults(t, configv1.KubeSchedulerConfiguration{
			Profiles: []configv1.KubeSchedulerProfile{{
				SchedulerName: ptr.To(v1.DefaultSchedulerName),
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
			Name:     "deferred-pod",
			NodeName: "node1",
			Priority: &lowPriority,
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

		// Verify the pod was enqueued automatically by the UpdatePod event handler!
		var found bool
		err = wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 2*time.Second, true, func(ctx context.Context) (bool, error) {
			_, found = queue.GetPod(pod.Name, pod.Namespace, nil)
			return found, nil
		})
		if err != nil || !found {
			t.Fatalf("Expected pod to be enqueued automatically by UpdatePod handler")
		}

		// Now update status to set PodResizePreemptionDisabled condition to True
		pod, err = cs.CoreV1().Pods(testCtx.NS.Name).Get(testCtx.Ctx, pod.Name, metav1.GetOptions{})
		if err != nil {
			t.Fatalf("Failed to get pod: %v", err)
		}
		pod.Status.Conditions = append(pod.Status.Conditions, v1.PodCondition{
			Type:   v1.PodResizePreemptionDisabled,
			Status: v1.ConditionTrue,
		})
		pod, err = cs.CoreV1().Pods(testCtx.NS.Name).UpdateStatus(testCtx.Ctx, pod, metav1.UpdateOptions{})
		if err != nil {
			t.Fatalf("Failed to set preemption disabled condition: %v", err)
		}

		// Verify the pod was removed from the queue!
		err = wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 2*time.Second, true, func(ctx context.Context) (bool, error) {
			_, found = queue.GetPod(pod.Name, pod.Namespace, nil)
			return !found, nil
		})
		if err != nil {
			t.Fatalf("Expected pod to be removed from the scheduling queue when preemption is disabled")
		}

		testutils.CleanupPods(testCtx.Ctx, cs, t, []*v1.Pod{pod})
	})

	t.Run("UpdatePod handler enqueues pod when PodResizePreemptionDisabled transitions from True to False", func(t *testing.T) {
		featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
			features.InPlacePodVerticalScaling:                    true,
			features.InPlacePodVerticalScalingSchedulerPreemption: true,
		})

		cfg := configtesting.V1ToInternalWithDefaults(t, configv1.KubeSchedulerConfiguration{
			Profiles: []configv1.KubeSchedulerProfile{{
				SchedulerName: ptr.To(v1.DefaultSchedulerName),
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

		// Create node
		nodeObject := st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{
			v1.ResourcePods: "32",
			v1.ResourceCPU:  "200m",
		}).Obj()
		if _, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, nodeObject, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create node: %v", err)
		}

		// Create a deferred resize pod assigned to node1 with preemption disabled (not enqueued initially)
		pod := initPausePod(&testutils.PausePodConfig{
			Name:     "deferred-pod",
			NodeName: "node1",
			Priority: &lowPriority,
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
			{
				Type:   v1.PodResizePreemptionDisabled,
				Status: v1.ConditionTrue,
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

		// Verify the pod is NOT in the queue
		time.Sleep(1 * time.Second)
		if _, found := queue.GetPod(pod.Name, pod.Namespace, nil); found {
			t.Fatalf("Expected pod to not be in scheduling queue because preemption is disabled")
		}

		// Now clear/remove the PodResizePreemptionDisabled condition
		pod, err = cs.CoreV1().Pods(testCtx.NS.Name).Get(testCtx.Ctx, pod.Name, metav1.GetOptions{})
		if err != nil {
			t.Fatalf("Failed to get pod: %v", err)
		}
		newConditions := []v1.PodCondition{}
		for _, cond := range pod.Status.Conditions {
			if cond.Type != v1.PodResizePreemptionDisabled {
				newConditions = append(newConditions, cond)
			}
		}
		pod.Status.Conditions = newConditions
		pod, err = cs.CoreV1().Pods(testCtx.NS.Name).UpdateStatus(testCtx.Ctx, pod, metav1.UpdateOptions{})
		if err != nil {
			t.Fatalf("Failed to clear preemption disabled condition: %v", err)
		}

		// Verify the pod was enqueued automatically by the UpdatePod event handler!
		var found bool
		err = wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 2*time.Second, true, func(ctx context.Context) (bool, error) {
			_, found = queue.GetPod(pod.Name, pod.Namespace, nil)
			return found, nil
		})
		if err != nil || !found {
			t.Fatalf("Expected pod to be enqueued automatically by UpdatePod handler on preemption enablement")
		}

		testutils.CleanupPods(testCtx.Ctx, cs, t, []*v1.Pod{pod})
	})
}
