// +build integration,!no-etcd

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
	"fmt"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/client/restclient"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/plugin/pkg/scheduler"
	_ "k8s.io/kubernetes/plugin/pkg/scheduler/algorithmprovider"
	"k8s.io/kubernetes/plugin/pkg/scheduler/factory"
	e2e "k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/integration/framework"
)

type nodeMutationFunc func(t *testing.T, n *api.Node, nodeStore cache.Store, c *client.Client)

type nodeStateManager struct {
	makeSchedulable   nodeMutationFunc
	makeUnSchedulable nodeMutationFunc
}

func TestUnschedulableNodes(t *testing.T) {
	_, s := framework.RunAMaster(nil)
	defer s.Close()

	ns := framework.CreateTestingNamespace("unschedulable-nodes", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	restClient := client.NewOrDie(&restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}})

	schedulerConfigFactory := factory.NewConfigFactory(restClient, api.DefaultSchedulerName, api.DefaultHardPodAffinitySymmetricWeight, api.DefaultFailureDomains)
	schedulerConfig, err := schedulerConfigFactory.Create()
	if err != nil {
		t.Fatalf("Couldn't create scheduler config: %v", err)
	}
	eventBroadcaster := record.NewBroadcaster()
	schedulerConfig.Recorder = eventBroadcaster.NewRecorder(api.EventSource{Component: api.DefaultSchedulerName})
	eventBroadcaster.StartRecordingToSink(restClient.Events(ns.Name))
	scheduler.New(schedulerConfig).Run()

	defer close(schedulerConfig.StopEverything)

	DoTestUnschedulableNodes(t, restClient, ns, schedulerConfigFactory.NodeLister.Store)
}

func podScheduled(c *client.Client, podNamespace, podName string) wait.ConditionFunc {
	return func() (bool, error) {
		pod, err := c.Pods(podNamespace).Get(podName)
		if errors.IsNotFound(err) {
			return false, nil
		}
		if err != nil {
			// This could be a connection error so we want to retry.
			return false, nil
		}
		if pod.Spec.NodeName == "" {
			return false, nil
		}
		return true, nil
	}
}

// Wait till the passFunc confirms that the object it expects to see is in the store.
// Used to observe reflected events.
func waitForReflection(t *testing.T, s cache.Store, key string, passFunc func(n interface{}) bool) error {
	nodes := []*api.Node{}
	err := wait.Poll(time.Millisecond*100, wait.ForeverTestTimeout, func() (bool, error) {
		if n, _, err := s.GetByKey(key); err == nil && passFunc(n) {
			return true, nil
		} else {
			if err != nil {
				t.Errorf("Unexpected error: %v", err)
			} else {
				if n == nil {
					nodes = append(nodes, nil)
				} else {
					nodes = append(nodes, n.(*api.Node))
				}
			}
			return false, nil
		}
	})
	if err != nil {
		t.Logf("Logging consecutive node versions received from store:")
		for i, n := range nodes {
			t.Logf("%d: %#v", i, n)
		}
	}
	return err
}

func DoTestUnschedulableNodes(t *testing.T, restClient *client.Client, ns *api.Namespace, nodeStore cache.Store) {
	// NOTE: This test cannot run in parallel, because it is creating and deleting
	// non-namespaced objects (Nodes).
	defer restClient.Nodes().DeleteCollection(nil, api.ListOptions{})

	goodCondition := api.NodeCondition{
		Type:              api.NodeReady,
		Status:            api.ConditionTrue,
		Reason:            fmt.Sprintf("schedulable condition"),
		LastHeartbeatTime: unversioned.Time{time.Now()},
	}
	badCondition := api.NodeCondition{
		Type:              api.NodeReady,
		Status:            api.ConditionUnknown,
		Reason:            fmt.Sprintf("unschedulable condition"),
		LastHeartbeatTime: unversioned.Time{time.Now()},
	}
	// Create a new schedulable node, since we're first going to apply
	// the unschedulable condition and verify that pods aren't scheduled.
	node := &api.Node{
		ObjectMeta: api.ObjectMeta{Name: "node-scheduling-test-node"},
		Spec:       api.NodeSpec{Unschedulable: false},
		Status: api.NodeStatus{
			Capacity: api.ResourceList{
				api.ResourcePods: *resource.NewQuantity(32, resource.DecimalSI),
			},
			Conditions: []api.NodeCondition{goodCondition},
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
			makeUnSchedulable: func(t *testing.T, n *api.Node, s cache.Store, c *client.Client) {
				n.Spec.Unschedulable = true
				if _, err := c.Nodes().Update(n); err != nil {
					t.Fatalf("Failed to update node with unschedulable=true: %v", err)
				}
				err = waitForReflection(t, s, nodeKey, func(node interface{}) bool {
					// An unschedulable node should still be present in the store
					// Nodes that are unschedulable or that are not ready or
					// have their disk full (Node.Spec.Conditions) are exluded
					// based on NodeConditionPredicate, a separate check
					return node != nil && node.(*api.Node).Spec.Unschedulable == true
				})
				if err != nil {
					t.Fatalf("Failed to observe reflected update for setting unschedulable=true: %v", err)
				}
			},
			makeSchedulable: func(t *testing.T, n *api.Node, s cache.Store, c *client.Client) {
				n.Spec.Unschedulable = false
				if _, err := c.Nodes().Update(n); err != nil {
					t.Fatalf("Failed to update node with unschedulable=false: %v", err)
				}
				err = waitForReflection(t, s, nodeKey, func(node interface{}) bool {
					return node != nil && node.(*api.Node).Spec.Unschedulable == false
				})
				if err != nil {
					t.Fatalf("Failed to observe reflected update for setting unschedulable=false: %v", err)
				}
			},
		},
		// Test node.Status.Conditions=ConditionTrue/Unknown
		{
			makeUnSchedulable: func(t *testing.T, n *api.Node, s cache.Store, c *client.Client) {
				n.Status = api.NodeStatus{
					Capacity: api.ResourceList{
						api.ResourcePods: *resource.NewQuantity(32, resource.DecimalSI),
					},
					Conditions: []api.NodeCondition{badCondition},
				}
				if _, err = c.Nodes().UpdateStatus(n); err != nil {
					t.Fatalf("Failed to update node with bad status condition: %v", err)
				}
				err = waitForReflection(t, s, nodeKey, func(node interface{}) bool {
					return node != nil && node.(*api.Node).Status.Conditions[0].Status == api.ConditionUnknown
				})
				if err != nil {
					t.Fatalf("Failed to observe reflected update for status condition update: %v", err)
				}
			},
			makeSchedulable: func(t *testing.T, n *api.Node, s cache.Store, c *client.Client) {
				n.Status = api.NodeStatus{
					Capacity: api.ResourceList{
						api.ResourcePods: *resource.NewQuantity(32, resource.DecimalSI),
					},
					Conditions: []api.NodeCondition{goodCondition},
				}
				if _, err = c.Nodes().UpdateStatus(n); err != nil {
					t.Fatalf("Failed to update node with healthy status condition: %v", err)
				}
				err = waitForReflection(t, s, nodeKey, func(node interface{}) bool {
					return node != nil && node.(*api.Node).Status.Conditions[0].Status == api.ConditionTrue
				})
				if err != nil {
					t.Fatalf("Failed to observe reflected update for status condition update: %v", err)
				}
			},
		},
	}

	for i, mod := range nodeModifications {
		unSchedNode, err := restClient.Nodes().Create(node)
		if err != nil {
			t.Fatalf("Failed to create node: %v", err)
		}

		// Apply the unschedulable modification to the node, and wait for the reflection
		mod.makeUnSchedulable(t, unSchedNode, nodeStore, restClient)

		// Create the new pod, note that this needs to happen post unschedulable
		// modification or we have a race in the test.
		pod := &api.Pod{
			ObjectMeta: api.ObjectMeta{Name: "node-scheduling-test-pod"},
			Spec: api.PodSpec{
				Containers: []api.Container{{Name: "container", Image: e2e.GetPauseImageName(restClient)}},
			},
		}
		myPod, err := restClient.Pods(ns.Name).Create(pod)
		if err != nil {
			t.Fatalf("Failed to create pod: %v", err)
		}

		// There are no schedulable nodes - the pod shouldn't be scheduled.
		err = wait.Poll(time.Second, wait.ForeverTestTimeout, podScheduled(restClient, myPod.Namespace, myPod.Name))
		if err == nil {
			t.Errorf("Pod scheduled successfully on unschedulable nodes")
		}
		if err != wait.ErrWaitTimeout {
			t.Errorf("Test %d: failed while trying to confirm the pod does not get scheduled on the node: %v", i, err)
		} else {
			t.Logf("Test %d: Pod did not get scheduled on an unschedulable node", i)
		}

		// Apply the schedulable modification to the node, and wait for the reflection
		schedNode, err := restClient.Nodes().Get(unSchedNode.Name)
		if err != nil {
			t.Fatalf("Failed to get node: %v", err)
		}
		mod.makeSchedulable(t, schedNode, nodeStore, restClient)

		// Wait until the pod is scheduled.
		err = wait.Poll(time.Second, wait.ForeverTestTimeout, podScheduled(restClient, myPod.Namespace, myPod.Name))
		if err != nil {
			t.Errorf("Test %d: failed to schedule a pod: %v", i, err)
		} else {
			t.Logf("Test %d: Pod got scheduled on a schedulable node", i)
		}

		err = restClient.Pods(ns.Name).Delete(myPod.Name, api.NewDeleteOptions(0))
		if err != nil {
			t.Errorf("Failed to delete pod: %v", err)
		}
		err = restClient.Nodes().Delete(schedNode.Name)
		if err != nil {
			t.Errorf("Failed to delete node: %v", err)
		}
	}
}

func TestMultiScheduler(t *testing.T) {
	_, s := framework.RunAMaster(nil)
	// TODO: Uncomment when fix #19254
	// This seems to be a different issue - it still doesn't work.
	// defer s.Close()

	ns := framework.CreateTestingNamespace("multi-scheduler", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	/*
		This integration tests the multi-scheduler feature in the following way:
		1. create a default scheduler
		2. create a node
		3. create 3 pods: testPodNoAnnotation, testPodWithAnnotationFitsDefault and testPodWithAnnotationFitsFoo
			- note: the first two should be picked and scheduled by default scheduler while the last one should be
			        picked by scheduler of name "foo-scheduler" which does not exist yet.
		4. **check point-1**:
			- testPodNoAnnotation, testPodWithAnnotationFitsDefault should be scheduled
			- testPodWithAnnotationFitsFoo should NOT be scheduled
		5. create a scheduler with name "foo-scheduler"
		6. **check point-2**:
			- testPodWithAnnotationFitsFoo should be scheduled
		7. stop default scheduler
		8. create 2 pods: testPodNoAnnotation2 and testPodWithAnnotationFitsDefault2
			- note: these two pods belong to default scheduler which no longer exists
		9. **check point-3**:
			- testPodNoAnnotation2 and testPodWithAnnotationFitsDefault2 shoule NOT be scheduled
	*/
	// 1. create and start default-scheduler
	restClient := client.NewOrDie(&restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}})

	// NOTE: This test cannot run in parallel, because it is creating and deleting
	// non-namespaced objects (Nodes).
	defer restClient.Nodes().DeleteCollection(nil, api.ListOptions{})

	schedulerConfigFactory := factory.NewConfigFactory(restClient, api.DefaultSchedulerName, api.DefaultHardPodAffinitySymmetricWeight, api.DefaultFailureDomains)
	schedulerConfig, err := schedulerConfigFactory.Create()
	if err != nil {
		t.Fatalf("Couldn't create scheduler config: %v", err)
	}
	eventBroadcaster := record.NewBroadcaster()
	schedulerConfig.Recorder = eventBroadcaster.NewRecorder(api.EventSource{Component: api.DefaultSchedulerName})
	eventBroadcaster.StartRecordingToSink(restClient.Events(ns.Name))
	scheduler.New(schedulerConfig).Run()
	// default-scheduler will be stopped later

	// 2. create a node
	node := &api.Node{
		ObjectMeta: api.ObjectMeta{Name: "node-multi-scheduler-test-node"},
		Spec:       api.NodeSpec{Unschedulable: false},
		Status: api.NodeStatus{
			Capacity: api.ResourceList{
				api.ResourcePods: *resource.NewQuantity(32, resource.DecimalSI),
			},
		},
	}
	restClient.Nodes().Create(node)

	// 3. create 3 pods for testing
	podWithNoAnnotation := createPod(restClient, "pod-with-no-annotation", nil)
	testPodNoAnnotation, err := restClient.Pods(ns.Name).Create(podWithNoAnnotation)
	if err != nil {
		t.Fatalf("Failed to create pod: %v", err)
	}

	schedulerAnnotationFitsDefault := map[string]string{"scheduler.alpha.kubernetes.io/name": "default-scheduler"}
	podWithAnnotationFitsDefault := createPod(restClient, "pod-with-annotation-fits-default", schedulerAnnotationFitsDefault)
	testPodWithAnnotationFitsDefault, err := restClient.Pods(ns.Name).Create(podWithAnnotationFitsDefault)
	if err != nil {
		t.Fatalf("Failed to create pod: %v", err)
	}

	schedulerAnnotationFitsFoo := map[string]string{"scheduler.alpha.kubernetes.io/name": "foo-scheduler"}
	podWithAnnotationFitsFoo := createPod(restClient, "pod-with-annotation-fits-foo", schedulerAnnotationFitsFoo)
	testPodWithAnnotationFitsFoo, err := restClient.Pods(ns.Name).Create(podWithAnnotationFitsFoo)
	if err != nil {
		t.Fatalf("Failed to create pod: %v", err)
	}

	// 4. **check point-1**:
	//		- testPodNoAnnotation, testPodWithAnnotationFitsDefault should be scheduled
	//		- testPodWithAnnotationFitsFoo should NOT be scheduled
	err = wait.Poll(time.Second, time.Second*5, podScheduled(restClient, testPodNoAnnotation.Namespace, testPodNoAnnotation.Name))
	if err != nil {
		t.Errorf("Test MultiScheduler: %s Pod not scheduled: %v", testPodNoAnnotation.Name, err)
	} else {
		t.Logf("Test MultiScheduler: %s Pod scheduled", testPodNoAnnotation.Name)
	}

	err = wait.Poll(time.Second, time.Second*5, podScheduled(restClient, testPodWithAnnotationFitsDefault.Namespace, testPodWithAnnotationFitsDefault.Name))
	if err != nil {
		t.Errorf("Test MultiScheduler: %s Pod not scheduled: %v", testPodWithAnnotationFitsDefault.Name, err)
	} else {
		t.Logf("Test MultiScheduler: %s Pod scheduled", testPodWithAnnotationFitsDefault.Name)
	}

	err = wait.Poll(time.Second, time.Second*5, podScheduled(restClient, testPodWithAnnotationFitsFoo.Namespace, testPodWithAnnotationFitsFoo.Name))
	if err == nil {
		t.Errorf("Test MultiScheduler: %s Pod got scheduled, %v", testPodWithAnnotationFitsFoo.Name, err)
	} else {
		t.Logf("Test MultiScheduler: %s Pod not scheduled", testPodWithAnnotationFitsFoo.Name)
	}

	// 5. create and start a scheduler with name "foo-scheduler"
	restClient2 := client.NewOrDie(&restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}})

	schedulerConfigFactory2 := factory.NewConfigFactory(restClient2, "foo-scheduler", api.DefaultHardPodAffinitySymmetricWeight, api.DefaultFailureDomains)
	schedulerConfig2, err := schedulerConfigFactory2.Create()
	if err != nil {
		t.Errorf("Couldn't create scheduler config: %v", err)
	}
	eventBroadcaster2 := record.NewBroadcaster()
	schedulerConfig2.Recorder = eventBroadcaster2.NewRecorder(api.EventSource{Component: "foo-scheduler"})
	eventBroadcaster2.StartRecordingToSink(restClient2.Events(ns.Name))
	scheduler.New(schedulerConfig2).Run()

	defer close(schedulerConfig2.StopEverything)

	//	6. **check point-2**:
	//		- testPodWithAnnotationFitsFoo should be scheduled
	err = wait.Poll(time.Second, time.Second*5, podScheduled(restClient, testPodWithAnnotationFitsFoo.Namespace, testPodWithAnnotationFitsFoo.Name))
	if err != nil {
		t.Errorf("Test MultiScheduler: %s Pod not scheduled, %v", testPodWithAnnotationFitsFoo.Name, err)
	} else {
		t.Logf("Test MultiScheduler: %s Pod scheduled", testPodWithAnnotationFitsFoo.Name)
	}

	//	7. delete the pods that were scheduled by the default scheduler, and stop the default scheduler
	err = restClient.Pods(ns.Name).Delete(testPodNoAnnotation.Name, api.NewDeleteOptions(0))
	if err != nil {
		t.Errorf("Failed to delete pod: %v", err)
	}
	err = restClient.Pods(ns.Name).Delete(testPodWithAnnotationFitsDefault.Name, api.NewDeleteOptions(0))
	if err != nil {
		t.Errorf("Failed to delete pod: %v", err)
	}

	// The rest of this test assumes that closing StopEverything will cause the
	// scheduler thread to stop immediately.  It won't, and in fact it will often
	// schedule 1 more pod before finally exiting.  Comment out until we fix that.
	//
	// See https://github.com/kubernetes/kubernetes/issues/23715 for more details.

	/*
		close(schedulerConfig.StopEverything)

		//	8. create 2 pods: testPodNoAnnotation2 and testPodWithAnnotationFitsDefault2
		//		- note: these two pods belong to default scheduler which no longer exists
		podWithNoAnnotation2 := createPod("pod-with-no-annotation2", nil)
		podWithAnnotationFitsDefault2 := createPod("pod-with-annotation-fits-default2", schedulerAnnotationFitsDefault)
		testPodNoAnnotation2, err := restClient.Pods(ns.Name).Create(podWithNoAnnotation2)
		if err != nil {
			t.Fatalf("Failed to create pod: %v", err)
		}
		testPodWithAnnotationFitsDefault2, err := restClient.Pods(ns.Name).Create(podWithAnnotationFitsDefault2)
		if err != nil {
			t.Fatalf("Failed to create pod: %v", err)
		}

		//	9. **check point-3**:
		//		- testPodNoAnnotation2 and testPodWithAnnotationFitsDefault2 shoule NOT be scheduled
		err = wait.Poll(time.Second, time.Second*5, podScheduled(restClient, testPodNoAnnotation2.Namespace, testPodNoAnnotation2.Name))
		if err == nil {
			t.Errorf("Test MultiScheduler: %s Pod got scheduled, %v", testPodNoAnnotation2.Name, err)
		} else {
			t.Logf("Test MultiScheduler: %s Pod not scheduled", testPodNoAnnotation2.Name)
		}
		err = wait.Poll(time.Second, time.Second*5, podScheduled(restClient, testPodWithAnnotationFitsDefault2.Namespace, testPodWithAnnotationFitsDefault2.Name))
		if err == nil {
			t.Errorf("Test MultiScheduler: %s Pod got scheduled, %v", testPodWithAnnotationFitsDefault2.Name, err)
		} else {
			t.Logf("Test MultiScheduler: %s Pod scheduled", testPodWithAnnotationFitsDefault2.Name)
		}
	*/
}

func createPod(client *client.Client, name string, annotation map[string]string) *api.Pod {
	return &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: name, Annotations: annotation},
		Spec: api.PodSpec{
			Containers: []api.Container{{Name: "container", Image: e2e.GetPauseImageName(client)}},
		},
	}
}

// This test will verify scheduler can work well regardless of whether kubelet is allocatable aware or not.
func TestAllocatable(t *testing.T) {
	_, s := framework.RunAMaster(nil)
	defer s.Close()

	ns := framework.CreateTestingNamespace("allocatable", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	// 1. create and start default-scheduler
	restClient := client.NewOrDie(&restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}})

	// NOTE: This test cannot run in parallel, because it is creating and deleting
	// non-namespaced objects (Nodes).
	defer restClient.Nodes().DeleteCollection(nil, api.ListOptions{})

	schedulerConfigFactory := factory.NewConfigFactory(restClient, api.DefaultSchedulerName, api.DefaultHardPodAffinitySymmetricWeight, api.DefaultFailureDomains)
	schedulerConfig, err := schedulerConfigFactory.Create()
	if err != nil {
		t.Fatalf("Couldn't create scheduler config: %v", err)
	}
	eventBroadcaster := record.NewBroadcaster()
	schedulerConfig.Recorder = eventBroadcaster.NewRecorder(api.EventSource{Component: api.DefaultSchedulerName})
	eventBroadcaster.StartRecordingToSink(restClient.Events(ns.Name))
	scheduler.New(schedulerConfig).Run()
	// default-scheduler will be stopped later
	defer close(schedulerConfig.StopEverything)

	// 2. create a node without allocatable awareness
	node := &api.Node{
		ObjectMeta: api.ObjectMeta{Name: "node-allocatable-scheduler-test-node"},
		Spec:       api.NodeSpec{Unschedulable: false},
		Status: api.NodeStatus{
			Capacity: api.ResourceList{
				api.ResourcePods:   *resource.NewQuantity(32, resource.DecimalSI),
				api.ResourceCPU:    *resource.NewMilliQuantity(30, resource.DecimalSI),
				api.ResourceMemory: *resource.NewQuantity(30, resource.BinarySI),
			},
		},
	}

	allocNode, err := restClient.Nodes().Create(node)
	if err != nil {
		t.Fatalf("Failed to create node: %v", err)
	}

	// 3. create resource pod which requires less than Capacity
	podResource := &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "pod-test-allocatable"},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:  "container",
					Image: e2e.GetPauseImageName(restClient),
					Resources: api.ResourceRequirements{
						Requests: api.ResourceList{
							api.ResourceCPU:    *resource.NewMilliQuantity(20, resource.DecimalSI),
							api.ResourceMemory: *resource.NewQuantity(20, resource.BinarySI),
						},
					},
				},
			},
		},
	}

	testAllocPod, err := restClient.Pods(ns.Name).Create(podResource)
	if err != nil {
		t.Fatalf("Test allocatable unawareness failed to create pod: %v", err)
	}

	// 4. Test: this test pod should be scheduled since api-server will use Capacity as Allocatable
	err = wait.Poll(time.Second, time.Second*5, podScheduled(restClient, testAllocPod.Namespace, testAllocPod.Name))
	if err != nil {
		t.Errorf("Test allocatable unawareness: %s Pod not scheduled: %v", testAllocPod.Name, err)
	} else {
		t.Logf("Test allocatable unawareness: %s Pod scheduled", testAllocPod.Name)
	}

	// 5. Change the node status to allocatable aware, note that Allocatable is less than Pod's requirement
	allocNode.Status = api.NodeStatus{
		Capacity: api.ResourceList{
			api.ResourcePods:   *resource.NewQuantity(32, resource.DecimalSI),
			api.ResourceCPU:    *resource.NewMilliQuantity(30, resource.DecimalSI),
			api.ResourceMemory: *resource.NewQuantity(30, resource.BinarySI),
		},
		Allocatable: api.ResourceList{
			api.ResourcePods:   *resource.NewQuantity(32, resource.DecimalSI),
			api.ResourceCPU:    *resource.NewMilliQuantity(10, resource.DecimalSI),
			api.ResourceMemory: *resource.NewQuantity(10, resource.BinarySI),
		},
	}

	if _, err := restClient.Nodes().UpdateStatus(allocNode); err != nil {
		t.Fatalf("Failed to update node with Status.Allocatable: %v", err)
	}

	if err := restClient.Pods(ns.Name).Delete(podResource.Name, &api.DeleteOptions{}); err != nil {
		t.Fatalf("Failed to remove first resource pod: %v", err)
	}

	// 6. Make another pod with different name, same resource request
	podResource.ObjectMeta.Name = "pod-test-allocatable2"
	testAllocPod2, err := restClient.Pods(ns.Name).Create(podResource)
	if err != nil {
		t.Fatalf("Test allocatable awareness failed to create pod: %v", err)
	}

	// 7. Test: this test pod should not be scheduled since it request more than Allocatable
	err = wait.Poll(time.Second, time.Second*5, podScheduled(restClient, testAllocPod2.Namespace, testAllocPod2.Name))
	if err == nil {
		t.Errorf("Test allocatable awareness: %s Pod got scheduled unexpectly, %v", testAllocPod2.Name, err)
	} else {
		t.Logf("Test allocatable awareness: %s Pod not scheduled as expected", testAllocPod2.Name)
	}
}
