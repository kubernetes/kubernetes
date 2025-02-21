/*
Copyright 2015 The Kubernetes Authors.

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

// This file tests the scheduler.

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	clientset "k8s.io/client-go/kubernetes"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	configv1 "k8s.io/kube-scheduler/config/v1"
	"k8s.io/kubernetes/pkg/scheduler"
	configtesting "k8s.io/kubernetes/pkg/scheduler/apis/config/testing"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	testutils "k8s.io/kubernetes/test/integration/util"
	"k8s.io/utils/pointer"
)

type nodeMutationFunc func(t *testing.T, n *v1.Node, nodeLister corelisters.NodeLister, c clientset.Interface)

type nodeStateManager struct {
	makeSchedulable   nodeMutationFunc
	makeUnSchedulable nodeMutationFunc
}

func TestUnschedulableNodes(t *testing.T) {
	testCtx := testutils.InitTestSchedulerWithNS(t, "unschedulable-nodes")

	nodeLister := testCtx.InformerFactory.Core().V1().Nodes().Lister()
	// NOTE: This test cannot run in parallel, because it is creating and deleting
	// non-namespaced objects (Nodes).
	defer testCtx.ClientSet.CoreV1().Nodes().DeleteCollection(context.TODO(), metav1.DeleteOptions{}, metav1.ListOptions{})

	goodCondition := v1.NodeCondition{
		Type:              v1.NodeReady,
		Status:            v1.ConditionTrue,
		Reason:            fmt.Sprintf("schedulable condition"),
		LastHeartbeatTime: metav1.Time{Time: time.Now()},
	}
	// Create a new schedulable node, since we're first going to apply
	// the unschedulable condition and verify that pods aren't scheduled.
	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: "node-scheduling-test-node"},
		Spec:       v1.NodeSpec{Unschedulable: false},
		Status: v1.NodeStatus{
			Capacity: v1.ResourceList{
				v1.ResourcePods: *resource.NewQuantity(32, resource.DecimalSI),
			},
			Conditions: []v1.NodeCondition{goodCondition},
		},
	}
	nodeKey, err := cache.MetaNamespaceKeyFunc(node)
	if err != nil {
		t.Fatalf("Couldn't retrieve key for node %v", node.Name)
	}

	// The test does the following for each nodeStateManager in this list:
	//	1. Create a new node
	//	2. Apply the makeUnSchedulable function
	//	3. Create a new pod
	//  4. Check that the pod doesn't get assigned to the node
	//  5. Apply the schedulable function
	//  6. Check that the pod *does* get assigned to the node
	//  7. Delete the pod and node.

	nodeModifications := []nodeStateManager{
		// Test node.Spec.Unschedulable=true/false
		{
			makeUnSchedulable: func(t *testing.T, n *v1.Node, nodeLister corelisters.NodeLister, c clientset.Interface) {
				n.Spec.Unschedulable = true
				if _, err := c.CoreV1().Nodes().Update(context.TODO(), n, metav1.UpdateOptions{}); err != nil {
					t.Fatalf("Failed to update node with unschedulable=true: %v", err)
				}
				err = testutils.WaitForReflection(testCtx.Ctx, t, nodeLister, nodeKey, func(node interface{}) bool {
					// An unschedulable node should still be present in the store
					// Nodes that are unschedulable or that are not ready or
					// have their disk full (Node.Spec.Conditions) are excluded
					// based on NodeConditionPredicate, a separate check
					return node != nil && node.(*v1.Node).Spec.Unschedulable
				})
				if err != nil {
					t.Fatalf("Failed to observe reflected update for setting unschedulable=true: %v", err)
				}
			},
			makeSchedulable: func(t *testing.T, n *v1.Node, nodeLister corelisters.NodeLister, c clientset.Interface) {
				n.Spec.Unschedulable = false
				if _, err := c.CoreV1().Nodes().Update(context.TODO(), n, metav1.UpdateOptions{}); err != nil {
					t.Fatalf("Failed to update node with unschedulable=false: %v", err)
				}
				err = testutils.WaitForReflection(testCtx.Ctx, t, nodeLister, nodeKey, func(node interface{}) bool {
					return node != nil && node.(*v1.Node).Spec.Unschedulable == false
				})
				if err != nil {
					t.Fatalf("Failed to observe reflected update for setting unschedulable=false: %v", err)
				}
			},
		},
	}

	for i, mod := range nodeModifications {
		unSchedNode, err := testutils.CreateNode(testCtx.ClientSet, node)
		if err != nil {
			t.Fatalf("Failed to create node: %v", err)
		}

		// Apply the unschedulable modification to the node, and wait for the reflection
		mod.makeUnSchedulable(t, unSchedNode, nodeLister, testCtx.ClientSet)

		// Create the new pod, note that this needs to happen post unschedulable
		// modification or we have a race in the test.
		myPod, err := testutils.CreatePausePodWithResource(testCtx.ClientSet, "node-scheduling-test-pod", testCtx.NS.Name, nil)
		if err != nil {
			t.Fatalf("Failed to create pod: %v", err)
		}

		// There are no schedulable nodes - the pod shouldn't be scheduled.
		err = testutils.WaitForPodToScheduleWithTimeout(testCtx.Ctx, testCtx.ClientSet, myPod, 2*time.Second)
		if err == nil {
			t.Errorf("Test %d: Pod scheduled successfully on unschedulable nodes", i)
		}
		if !wait.Interrupted(err) {
			t.Errorf("Test %d: failed while trying to confirm the pod does not get scheduled on the node: %v", i, err)
		} else {
			t.Logf("Test %d: Pod did not get scheduled on an unschedulable node", i)
		}

		// Apply the schedulable modification to the node, and wait for the reflection
		schedNode, err := testCtx.ClientSet.CoreV1().Nodes().Get(context.TODO(), unSchedNode.Name, metav1.GetOptions{})
		if err != nil {
			t.Fatalf("Failed to get node: %v", err)
		}
		mod.makeSchedulable(t, schedNode, nodeLister, testCtx.ClientSet)

		// Wait until the pod is scheduled.
		if err := testutils.WaitForPodToSchedule(testCtx.Ctx, testCtx.ClientSet, myPod); err != nil {
			t.Errorf("Test %d: failed to schedule a pod: %v", i, err)
		} else {
			t.Logf("Test %d: Pod got scheduled on a schedulable node", i)
		}
		// Clean up.
		if err := testutils.DeletePod(testCtx.ClientSet, myPod.Name, myPod.Namespace); err != nil {
			t.Errorf("Failed to delete pod: %v", err)
		}
		err = testCtx.ClientSet.CoreV1().Nodes().Delete(context.TODO(), schedNode.Name, metav1.DeleteOptions{})
		if err != nil {
			t.Errorf("Failed to delete node: %v", err)
		}
	}
}

func TestMultipleSchedulers(t *testing.T) {
	// This integration tests the multi-scheduler feature in the following way:
	// 1. create a default scheduler
	// 2. create a node
	// 3. create 3 pods: testPodNoAnnotation, testPodWithAnnotationFitsDefault and testPodWithAnnotationFitsFoo
	//	  - note: the first two should be picked and scheduled by default scheduler while the last one should be
	//	          picked by scheduler of name "foo-scheduler" which does not exist yet.
	// 4. **check point-1**:
	//	   - testPodNoAnnotation, testPodWithAnnotationFitsDefault should be scheduled
	//	   - testPodWithAnnotationFitsFoo should NOT be scheduled
	// 5. create a scheduler with name "foo-scheduler"
	// 6. **check point-2**:
	//     - testPodWithAnnotationFitsFoo should be scheduled

	// 1. create and start default-scheduler
	testCtx := testutils.InitTestSchedulerWithNS(t, "multi-scheduler")

	// 2. create a node
	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: "node-multi-scheduler-test-node"},
		Spec:       v1.NodeSpec{Unschedulable: false},
		Status: v1.NodeStatus{
			Capacity: v1.ResourceList{
				v1.ResourcePods: *resource.NewQuantity(32, resource.DecimalSI),
			},
		},
	}
	testutils.CreateNode(testCtx.ClientSet, node)

	// 3. create 3 pods for testing
	t.Logf("create 3 pods for testing")
	testPod, err := testutils.CreatePausePodWithResource(testCtx.ClientSet, "pod-without-scheduler-name", testCtx.NS.Name, nil)
	if err != nil {
		t.Fatalf("Failed to create pod: %v", err)
	}

	defaultScheduler := "default-scheduler"
	testPodFitsDefault, err := testutils.CreatePausePod(testCtx.ClientSet, testutils.InitPausePod(&testutils.PausePodConfig{Name: "pod-fits-default", Namespace: testCtx.NS.Name, SchedulerName: defaultScheduler}))
	if err != nil {
		t.Fatalf("Failed to create pod: %v", err)
	}

	fooScheduler := "foo-scheduler"
	testPodFitsFoo, err := testutils.CreatePausePod(testCtx.ClientSet, testutils.InitPausePod(&testutils.PausePodConfig{Name: "pod-fits-foo", Namespace: testCtx.NS.Name, SchedulerName: fooScheduler}))
	if err != nil {
		t.Fatalf("Failed to create pod: %v", err)
	}

	// 4. **check point-1**:
	//		- testPod, testPodFitsDefault should be scheduled
	//		- testPodFitsFoo should NOT be scheduled
	t.Logf("wait for pods scheduled")
	if err := testutils.WaitForPodToSchedule(testCtx.Ctx, testCtx.ClientSet, testPod); err != nil {
		t.Errorf("Test MultiScheduler: %s Pod not scheduled: %v", testPod.Name, err)
	} else {
		t.Logf("Test MultiScheduler: %s Pod scheduled", testPod.Name)
	}

	if err := testutils.WaitForPodToSchedule(testCtx.Ctx, testCtx.ClientSet, testPodFitsDefault); err != nil {
		t.Errorf("Test MultiScheduler: %s Pod not scheduled: %v", testPodFitsDefault.Name, err)
	} else {
		t.Logf("Test MultiScheduler: %s Pod scheduled", testPodFitsDefault.Name)
	}

	if err := testutils.WaitForPodToScheduleWithTimeout(testCtx.Ctx, testCtx.ClientSet, testPodFitsFoo, time.Second*5); err == nil {
		t.Errorf("Test MultiScheduler: %s Pod got scheduled, %v", testPodFitsFoo.Name, err)
	} else {
		t.Logf("Test MultiScheduler: %s Pod not scheduled", testPodFitsFoo.Name)
	}

	// 5. create and start a scheduler with name "foo-scheduler"
	cfg := configtesting.V1ToInternalWithDefaults(t, configv1.KubeSchedulerConfiguration{
		Profiles: []configv1.KubeSchedulerProfile{{
			SchedulerName: pointer.String(fooScheduler),
			PluginConfig: []configv1.PluginConfig{
				{
					Name: "VolumeBinding",
					Args: runtime.RawExtension{
						Object: &configv1.VolumeBindingArgs{
							BindTimeoutSeconds: pointer.Int64(30),
						},
					},
				},
			}},
		},
	})
	testCtx = testutils.InitTestSchedulerWithOptions(t, testCtx, 0, scheduler.WithProfiles(cfg.Profiles...))
	testutils.SyncSchedulerInformerFactory(testCtx)
	go testCtx.Scheduler.Run(testCtx.Ctx)

	//	6. **check point-2**:
	//		- testPodWithAnnotationFitsFoo should be scheduled
	err = testutils.WaitForPodToSchedule(testCtx.Ctx, testCtx.ClientSet, testPodFitsFoo)
	if err != nil {
		t.Errorf("Test MultiScheduler: %s Pod not scheduled, %v", testPodFitsFoo.Name, err)
	} else {
		t.Logf("Test MultiScheduler: %s Pod scheduled", testPodFitsFoo.Name)
	}
}

func TestMultipleSchedulingProfiles(t *testing.T) {
	cfg := configtesting.V1ToInternalWithDefaults(t, configv1.KubeSchedulerConfiguration{
		Profiles: []configv1.KubeSchedulerProfile{
			{SchedulerName: pointer.String("default-scheduler")},
			{SchedulerName: pointer.String("custom-scheduler")},
		},
	})

	testCtx := testutils.InitTestSchedulerWithNS(t, "multi-scheduler", scheduler.WithProfiles(cfg.Profiles...))

	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: "node-multi-scheduler-test-node"},
		Spec:       v1.NodeSpec{Unschedulable: false},
		Status: v1.NodeStatus{
			Capacity: v1.ResourceList{
				v1.ResourcePods: *resource.NewQuantity(32, resource.DecimalSI),
			},
		},
	}
	if _, err := testutils.CreateNode(testCtx.ClientSet, node); err != nil {
		t.Fatal(err)
	}

	evs, err := testCtx.ClientSet.CoreV1().Events(testCtx.NS.Name).Watch(testCtx.Ctx, metav1.ListOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer evs.Stop()

	for _, pc := range []*testutils.PausePodConfig{
		{Name: "foo", Namespace: testCtx.NS.Name},
		{Name: "bar", Namespace: testCtx.NS.Name, SchedulerName: "unknown-scheduler"},
		{Name: "baz", Namespace: testCtx.NS.Name, SchedulerName: "default-scheduler"},
		{Name: "zet", Namespace: testCtx.NS.Name, SchedulerName: "custom-scheduler"},
	} {
		if _, err := testutils.CreatePausePod(testCtx.ClientSet, testutils.InitPausePod(pc)); err != nil {
			t.Fatal(err)
		}
	}

	wantProfiles := map[string]string{
		"foo": "default-scheduler",
		"baz": "default-scheduler",
		"zet": "custom-scheduler",
	}

	gotProfiles := make(map[string]string)
	if err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 30*time.Second, false, func(ctx context.Context) (bool, error) {
		var ev watch.Event
		select {
		case ev = <-evs.ResultChan():
		case <-time.After(30 * time.Second):
			return false, nil
		}
		e, ok := ev.Object.(*v1.Event)
		if !ok || e.Reason != "Scheduled" {
			return false, nil
		}
		gotProfiles[e.InvolvedObject.Name] = e.ReportingController
		return len(gotProfiles) >= len(wantProfiles), nil
	}); err != nil {
		t.Errorf("waiting for scheduling events: %v", err)
	}

	if diff := cmp.Diff(wantProfiles, gotProfiles); diff != "" {
		t.Errorf("pods scheduled by the wrong profile (-want, +got):\n%s", diff)
	}
}

// This test will verify scheduler can work well regardless of whether kubelet is allocatable aware or not.
func TestAllocatable(t *testing.T) {
	testCtx := testutils.InitTestSchedulerWithNS(t, "allocatable")

	// 2. create a node without allocatable awareness
	nodeRes := map[v1.ResourceName]string{
		v1.ResourcePods:   "32",
		v1.ResourceCPU:    "30m",
		v1.ResourceMemory: "30",
	}
	allocNode, err := testutils.CreateNode(testCtx.ClientSet, st.MakeNode().Name("node-allocatable-scheduler-test-node").Capacity(nodeRes).Obj())
	if err != nil {
		t.Fatalf("Failed to create node: %v", err)
	}

	// 3. create resource pod which requires less than Capacity
	podName := "pod-test-allocatable"
	podRes := &v1.ResourceList{
		v1.ResourceCPU:    *resource.NewMilliQuantity(20, resource.DecimalSI),
		v1.ResourceMemory: *resource.NewQuantity(20, resource.BinarySI),
	}
	testAllocPod, err := testutils.CreatePausePodWithResource(testCtx.ClientSet, podName, testCtx.NS.Name, podRes)
	if err != nil {
		t.Fatalf("Test allocatable unawareness failed to create pod: %v", err)
	}

	// 4. Test: this test pod should be scheduled since api-server will use Capacity as Allocatable
	err = testutils.WaitForPodToScheduleWithTimeout(testCtx.Ctx, testCtx.ClientSet, testAllocPod, time.Second*5)
	if err != nil {
		t.Errorf("Test allocatable unawareness: %s Pod not scheduled: %v", testAllocPod.Name, err)
	} else {
		t.Logf("Test allocatable unawareness: %s Pod scheduled", testAllocPod.Name)
	}

	// 5. Change the node status to allocatable aware, note that Allocatable is less than Pod's requirement
	allocNode.Status = v1.NodeStatus{
		Capacity: v1.ResourceList{
			v1.ResourcePods:   *resource.NewQuantity(32, resource.DecimalSI),
			v1.ResourceCPU:    *resource.NewMilliQuantity(30, resource.DecimalSI),
			v1.ResourceMemory: *resource.NewQuantity(30, resource.BinarySI),
		},
		Allocatable: v1.ResourceList{
			v1.ResourcePods:   *resource.NewQuantity(32, resource.DecimalSI),
			v1.ResourceCPU:    *resource.NewMilliQuantity(10, resource.DecimalSI),
			v1.ResourceMemory: *resource.NewQuantity(10, resource.BinarySI),
		},
	}

	if _, err := testCtx.ClientSet.CoreV1().Nodes().UpdateStatus(context.TODO(), allocNode, metav1.UpdateOptions{}); err != nil {
		t.Fatalf("Failed to update node with Status.Allocatable: %v", err)
	}

	if err := testutils.DeletePod(testCtx.ClientSet, testAllocPod.Name, testCtx.NS.Name); err != nil {
		t.Fatalf("Failed to remove the first pod: %v", err)
	}

	// 6. Make another pod with different name, same resource request
	podName2 := "pod-test-allocatable2"
	testAllocPod2, err := testutils.CreatePausePodWithResource(testCtx.ClientSet, podName2, testCtx.NS.Name, podRes)
	if err != nil {
		t.Fatalf("Test allocatable awareness failed to create pod: %v", err)
	}

	// 7. Test: this test pod should not be scheduled since it request more than Allocatable
	if err := testutils.WaitForPodToScheduleWithTimeout(testCtx.Ctx, testCtx.ClientSet, testAllocPod2, time.Second*5); err == nil {
		t.Errorf("Test allocatable awareness: %s Pod got scheduled unexpectedly, %v", testAllocPod2.Name, err)
	} else {
		t.Logf("Test allocatable awareness: %s Pod not scheduled as expected", testAllocPod2.Name)
	}
}

// TestSchedulerInformers tests that scheduler receives informer events and updates its cache when
// pods are scheduled by other schedulers.
func TestSchedulerInformers(t *testing.T) {
	// Initialize scheduler.
	testCtx := testutils.InitTestSchedulerWithNS(t, "scheduler-informer")
	cs := testCtx.ClientSet

	defaultPodRes := &v1.ResourceRequirements{Requests: v1.ResourceList{
		v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
		v1.ResourceMemory: *resource.NewQuantity(200, resource.BinarySI)},
	}
	defaultNodeRes := map[v1.ResourceName]string{
		v1.ResourcePods:   "32",
		v1.ResourceCPU:    "500m",
		v1.ResourceMemory: "500",
	}

	type nodeConfig struct {
		name string
		res  map[v1.ResourceName]string
	}

	tests := []struct {
		name                string
		nodes               []*nodeConfig
		existingPods        []*v1.Pod
		pod                 *v1.Pod
		preemptedPodIndexes map[int]struct{}
	}{
		{
			name:  "Pod cannot be scheduled when node is occupied by pods scheduled by other schedulers",
			nodes: []*nodeConfig{{name: "node-1", res: defaultNodeRes}},
			existingPods: []*v1.Pod{
				testutils.InitPausePod(&testutils.PausePodConfig{
					Name:          "pod1",
					Namespace:     testCtx.NS.Name,
					Resources:     defaultPodRes,
					Labels:        map[string]string{"foo": "bar"},
					NodeName:      "node-1",
					SchedulerName: "foo-scheduler",
				}),
				testutils.InitPausePod(&testutils.PausePodConfig{
					Name:          "pod2",
					Namespace:     testCtx.NS.Name,
					Resources:     defaultPodRes,
					Labels:        map[string]string{"foo": "bar"},
					NodeName:      "node-1",
					SchedulerName: "bar-scheduler",
				}),
			},
			pod: testutils.InitPausePod(&testutils.PausePodConfig{
				Name:      "unschedulable-pod",
				Namespace: testCtx.NS.Name,
				Resources: defaultPodRes,
			}),
			preemptedPodIndexes: map[int]struct{}{2: {}},
		},
		{
			name:         "The pod cannot be scheduled when nodeAffinity specifies a non-existent node.",
			nodes:        []*nodeConfig{{name: "node-1", res: defaultNodeRes}},
			existingPods: []*v1.Pod{},
			pod: testutils.InitPausePod(&testutils.PausePodConfig{
				Name:      "unschedulable-pod",
				Namespace: testCtx.NS.Name,
				Affinity: &v1.Affinity{
					NodeAffinity: &v1.NodeAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
							NodeSelectorTerms: []v1.NodeSelectorTerm{
								{
									MatchFields: []v1.NodeSelectorRequirement{
										{
											Key:      "metadata.name",
											Operator: v1.NodeSelectorOpIn,
											Values:   []string{"invalid-node"},
										},
									},
								},
							},
						},
					},
				},
				Resources: defaultPodRes,
			}),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			for _, nodeConf := range test.nodes {
				_, err := testutils.CreateNode(cs, st.MakeNode().Name(nodeConf.name).Capacity(nodeConf.res).Obj())
				if err != nil {
					t.Fatalf("Error creating node %v: %v", nodeConf.name, err)
				}
			}
			// Ensure nodes are present in scheduler cache.
			if err := testutils.WaitForNodesInCache(testCtx.Ctx, testCtx.Scheduler, len(test.nodes)); err != nil {
				t.Fatal(err)
			}

			pods := make([]*v1.Pod, len(test.existingPods))
			var err error
			// Create and run existingPods.
			for i, p := range test.existingPods {
				if pods[i], err = testutils.RunPausePod(cs, p); err != nil {
					t.Fatalf("Error running pause pod: %v", err)
				}
			}
			// Create the new "pod".
			unschedulable, err := testutils.CreatePausePod(cs, test.pod)
			if err != nil {
				t.Errorf("Error while creating new pod: %v", err)
			}
			if err := testutils.WaitForPodUnschedulable(testCtx.Ctx, cs, unschedulable); err != nil {
				t.Errorf("Pod %v got scheduled: %v", unschedulable.Name, err)
			}

			// Cleanup
			pods = append(pods, unschedulable)
			testutils.CleanupPods(testCtx.Ctx, cs, t, pods)
			if err := cs.PolicyV1().PodDisruptionBudgets(testCtx.NS.Name).DeleteCollection(testCtx.Ctx, metav1.DeleteOptions{}, metav1.ListOptions{}); err != nil {
				t.Errorf("error whiling deleting PDBs, error: %v", err)
			}
			if err := cs.CoreV1().Nodes().DeleteCollection(testCtx.Ctx, metav1.DeleteOptions{}, metav1.ListOptions{}); err != nil {
				t.Errorf("error whiling deleting nodes, error: %v", err)
			}
		})
	}
}

func TestNodeEvents(t *testing.T) {
	// The test verifies that unschedulable pods are re-queued
	// on node update events. The scenario we are testing is the following:
	// 1. Create pod1 and node1 that is small enough to only fit pod1; pod1 schedules on node1
	// 2. Create pod2, it should be unschedulable due to insufficient cpu
	// 3. Create node2 with a taint, pod2 should still not schedule
	// 4. Remove the taint from node2; pod2 should now schedule on node2

	testCtx := testutils.InitTestSchedulerWithNS(t, "node-events")

	// 1.1 create pod1
	pod1, err := testutils.CreatePausePodWithResource(testCtx.ClientSet, "pod1", testCtx.NS.Name, &v1.ResourceList{
		v1.ResourceCPU: *resource.NewMilliQuantity(80, resource.DecimalSI),
	})
	if err != nil {
		t.Fatalf("Failed to create pod: %v", err)
	}

	// 1.2 Create node1
	node1, err := testutils.CreateNode(testCtx.ClientSet, st.MakeNode().
		Name("node-events-test-node1").
		Capacity(map[v1.ResourceName]string{
			v1.ResourcePods:   "32",
			v1.ResourceCPU:    "100m",
			v1.ResourceMemory: "30",
		}).Obj())
	if err != nil {
		t.Fatalf("Failed to create %s: %v", node1.Name, err)
	}

	// 1.3 verify pod1 is scheduled
	err = testutils.WaitForPodToScheduleWithTimeout(testCtx.Ctx, testCtx.ClientSet, pod1, time.Second*5)
	if err != nil {
		t.Errorf("Pod %s didn't schedule: %v", pod1.Name, err)
	}

	// 2. create pod2
	pod2, err := testutils.CreatePausePodWithResource(testCtx.ClientSet, "pod2", testCtx.NS.Name, &v1.ResourceList{
		v1.ResourceCPU: *resource.NewMilliQuantity(40, resource.DecimalSI),
	})
	if err != nil {
		t.Fatalf("Failed to create pod %v: %v", pod2.Name, err)
	}

	if err := testutils.WaitForPodUnschedulable(testCtx.Ctx, testCtx.ClientSet, pod2); err != nil {
		t.Errorf("Pod %v got scheduled: %v", pod2.Name, err)
	}

	// 3.1 Create node2 with a taint
	node2 := st.MakeNode().
		Name("node-events-test-node2").
		Capacity(map[v1.ResourceName]string{
			v1.ResourcePods:   "32",
			v1.ResourceCPU:    "100m",
			v1.ResourceMemory: "30",
		}).
		Label("affinity-key", "affinity-value").
		Taints([]v1.Taint{{Key: "taint-key", Effect: v1.TaintEffectNoSchedule}}).Obj()
	node2, err = testutils.CreateNode(testCtx.ClientSet, node2)
	if err != nil {
		t.Fatalf("Failed to create %s: %v", node2.Name, err)
	}
	// make sure the scheduler received the node add event by creating a pod that only fits node2
	plugPod := st.MakePod().Name("plug-pod").Namespace(testCtx.NS.Name).Container("pause").
		Req(map[v1.ResourceName]string{v1.ResourceCPU: "40m"}).
		NodeAffinityIn("affinity-key", []string{"affinity-value"}, st.NodeSelectorTypeMatchExpressions).
		Toleration("taint-key").Obj()
	plugPod, err = testCtx.ClientSet.CoreV1().Pods(plugPod.Namespace).Create(testCtx.Ctx, plugPod, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create pod %v: %v", plugPod.Name, err)
	}
	err = testutils.WaitForPodToScheduleWithTimeout(testCtx.Ctx, testCtx.ClientSet, plugPod, time.Second*5)
	if err != nil {
		t.Errorf("Pod %s didn't schedule: %v", plugPod.Name, err)
	}

	// 3.2 pod2 still unschedulable
	if err := testutils.WaitForPodUnschedulable(testCtx.Ctx, testCtx.ClientSet, pod2); err != nil {
		t.Errorf("Pod %v got scheduled: %v", pod2.Name, err)
	}

	// 4. Remove node taint, pod2 should schedule
	node2.Spec.Taints = nil
	node2, err = testutils.UpdateNode(testCtx.ClientSet, node2)
	if err != nil {
		t.Fatalf("Failed to update %s: %v", node2.Name, err)
	}

	err = testutils.WaitForPodToScheduleWithTimeout(testCtx.Ctx, testCtx.ClientSet, pod2, time.Second*5)
	if err != nil {
		t.Errorf("Pod %s didn't schedule: %v", pod2.Name, err)
	}

}
