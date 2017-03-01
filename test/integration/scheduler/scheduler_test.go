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

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientv1core "k8s.io/client-go/kubernetes/typed/core/v1"
	clientv1 "k8s.io/client-go/pkg/api/v1"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	informers "k8s.io/kubernetes/pkg/client/informers/informers_generated/externalversions"
	corelisters "k8s.io/kubernetes/pkg/client/listers/core/v1"
	"k8s.io/kubernetes/plugin/pkg/scheduler"
	_ "k8s.io/kubernetes/plugin/pkg/scheduler/algorithmprovider"
	"k8s.io/kubernetes/plugin/pkg/scheduler/factory"
	e2e "k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/integration/framework"
)

type nodeMutationFunc func(t *testing.T, n *v1.Node, nodeLister corelisters.NodeLister, c clientset.Interface)

type nodeStateManager struct {
	makeSchedulable   nodeMutationFunc
	makeUnSchedulable nodeMutationFunc
}

func TestUnschedulableNodes(t *testing.T) {
	_, s := framework.RunAMaster(nil)
	defer s.Close()

	ns := framework.CreateTestingNamespace("unschedulable-nodes", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	clientSet := clientset.NewForConfigOrDie(&restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &api.Registry.GroupOrDie(v1.GroupName).GroupVersion}})
	informerFactory := informers.NewSharedInformerFactory(clientSet, 0)

	schedulerConfigFactory := factory.NewConfigFactory(
		v1.DefaultSchedulerName,
		clientSet,
		informerFactory.Core().V1().Nodes(),
		informerFactory.Core().V1().PersistentVolumes(),
		informerFactory.Core().V1().PersistentVolumeClaims(),
		informerFactory.Core().V1().ReplicationControllers(),
		informerFactory.Extensions().V1beta1().ReplicaSets(),
		informerFactory.Apps().V1beta1().StatefulSets(),
		informerFactory.Core().V1().Services(),
		v1.DefaultHardPodAffinitySymmetricWeight,
	)
	schedulerConfig, err := schedulerConfigFactory.Create()
	if err != nil {
		t.Fatalf("Couldn't create scheduler config: %v", err)
	}
	eventBroadcaster := record.NewBroadcaster()
	schedulerConfig.Recorder = eventBroadcaster.NewRecorder(api.Scheme, clientv1.EventSource{Component: v1.DefaultSchedulerName})
	eventBroadcaster.StartRecordingToSink(&clientv1core.EventSinkImpl{Interface: clientv1core.New(clientSet.Core().RESTClient()).Events("")})
	informerFactory.Start(schedulerConfig.StopEverything)
	scheduler.New(schedulerConfig).Run()

	defer close(schedulerConfig.StopEverything)

	DoTestUnschedulableNodes(t, clientSet, ns, schedulerConfigFactory.GetNodeLister())
}

func podScheduled(c clientset.Interface, podNamespace, podName string) wait.ConditionFunc {
	return func() (bool, error) {
		pod, err := c.Core().Pods(podNamespace).Get(podName, metav1.GetOptions{})
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
func waitForReflection(t *testing.T, nodeLister corelisters.NodeLister, key string, passFunc func(n interface{}) bool) error {
	nodes := []*v1.Node{}
	err := wait.Poll(time.Millisecond*100, wait.ForeverTestTimeout, func() (bool, error) {
		n, err := nodeLister.Get(key)

		switch {
		case err == nil && passFunc(n):
			return true, nil
		case errors.IsNotFound(err):
			nodes = append(nodes, nil)
		case err != nil:
			t.Errorf("Unexpected error: %v", err)
		default:
			nodes = append(nodes, n)
		}

		return false, nil
	})
	if err != nil {
		t.Logf("Logging consecutive node versions received from store:")
		for i, n := range nodes {
			t.Logf("%d: %#v", i, n)
		}
	}
	return err
}

func DoTestUnschedulableNodes(t *testing.T, cs clientset.Interface, ns *v1.Namespace, nodeLister corelisters.NodeLister) {
	// NOTE: This test cannot run in parallel, because it is creating and deleting
	// non-namespaced objects (Nodes).
	defer cs.Core().Nodes().DeleteCollection(nil, metav1.ListOptions{})

	goodCondition := v1.NodeCondition{
		Type:              v1.NodeReady,
		Status:            v1.ConditionTrue,
		Reason:            fmt.Sprintf("schedulable condition"),
		LastHeartbeatTime: metav1.Time{time.Now()},
	}
	badCondition := v1.NodeCondition{
		Type:              v1.NodeReady,
		Status:            v1.ConditionUnknown,
		Reason:            fmt.Sprintf("unschedulable condition"),
		LastHeartbeatTime: metav1.Time{time.Now()},
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
				if _, err := c.Core().Nodes().Update(n); err != nil {
					t.Fatalf("Failed to update node with unschedulable=true: %v", err)
				}
				err = waitForReflection(t, nodeLister, nodeKey, func(node interface{}) bool {
					// An unschedulable node should still be present in the store
					// Nodes that are unschedulable or that are not ready or
					// have their disk full (Node.Spec.Conditions) are excluded
					// based on NodeConditionPredicate, a separate check
					return node != nil && node.(*v1.Node).Spec.Unschedulable == true
				})
				if err != nil {
					t.Fatalf("Failed to observe reflected update for setting unschedulable=true: %v", err)
				}
			},
			makeSchedulable: func(t *testing.T, n *v1.Node, nodeLister corelisters.NodeLister, c clientset.Interface) {
				n.Spec.Unschedulable = false
				if _, err := c.Core().Nodes().Update(n); err != nil {
					t.Fatalf("Failed to update node with unschedulable=false: %v", err)
				}
				err = waitForReflection(t, nodeLister, nodeKey, func(node interface{}) bool {
					return node != nil && node.(*v1.Node).Spec.Unschedulable == false
				})
				if err != nil {
					t.Fatalf("Failed to observe reflected update for setting unschedulable=false: %v", err)
				}
			},
		},
		// Test node.Status.Conditions=ConditionTrue/Unknown
		{
			makeUnSchedulable: func(t *testing.T, n *v1.Node, nodeLister corelisters.NodeLister, c clientset.Interface) {
				n.Status = v1.NodeStatus{
					Capacity: v1.ResourceList{
						v1.ResourcePods: *resource.NewQuantity(32, resource.DecimalSI),
					},
					Conditions: []v1.NodeCondition{badCondition},
				}
				if _, err = c.Core().Nodes().UpdateStatus(n); err != nil {
					t.Fatalf("Failed to update node with bad status condition: %v", err)
				}
				err = waitForReflection(t, nodeLister, nodeKey, func(node interface{}) bool {
					return node != nil && node.(*v1.Node).Status.Conditions[0].Status == v1.ConditionUnknown
				})
				if err != nil {
					t.Fatalf("Failed to observe reflected update for status condition update: %v", err)
				}
			},
			makeSchedulable: func(t *testing.T, n *v1.Node, nodeLister corelisters.NodeLister, c clientset.Interface) {
				n.Status = v1.NodeStatus{
					Capacity: v1.ResourceList{
						v1.ResourcePods: *resource.NewQuantity(32, resource.DecimalSI),
					},
					Conditions: []v1.NodeCondition{goodCondition},
				}
				if _, err = c.Core().Nodes().UpdateStatus(n); err != nil {
					t.Fatalf("Failed to update node with healthy status condition: %v", err)
				}
				err = waitForReflection(t, nodeLister, nodeKey, func(node interface{}) bool {
					return node != nil && node.(*v1.Node).Status.Conditions[0].Status == v1.ConditionTrue
				})
				if err != nil {
					t.Fatalf("Failed to observe reflected update for status condition update: %v", err)
				}
			},
		},
	}

	for i, mod := range nodeModifications {
		unSchedNode, err := cs.Core().Nodes().Create(node)
		if err != nil {
			t.Fatalf("Failed to create node: %v", err)
		}

		// Apply the unschedulable modification to the node, and wait for the reflection
		mod.makeUnSchedulable(t, unSchedNode, nodeLister, cs)

		// Create the new pod, note that this needs to happen post unschedulable
		// modification or we have a race in the test.
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "node-scheduling-test-pod"},
			Spec: v1.PodSpec{
				Containers: []v1.Container{{Name: "container", Image: e2e.GetPauseImageName(cs)}},
			},
		}
		myPod, err := cs.Core().Pods(ns.Name).Create(pod)
		if err != nil {
			t.Fatalf("Failed to create pod: %v", err)
		}

		// There are no schedulable nodes - the pod shouldn't be scheduled.
		err = wait.Poll(time.Second, wait.ForeverTestTimeout, podScheduled(cs, myPod.Namespace, myPod.Name))
		if err == nil {
			t.Errorf("Pod scheduled successfully on unschedulable nodes")
		}
		if err != wait.ErrWaitTimeout {
			t.Errorf("Test %d: failed while trying to confirm the pod does not get scheduled on the node: %v", i, err)
		} else {
			t.Logf("Test %d: Pod did not get scheduled on an unschedulable node", i)
		}

		// Apply the schedulable modification to the node, and wait for the reflection
		schedNode, err := cs.Core().Nodes().Get(unSchedNode.Name, metav1.GetOptions{})
		if err != nil {
			t.Fatalf("Failed to get node: %v", err)
		}
		mod.makeSchedulable(t, schedNode, nodeLister, cs)

		// Wait until the pod is scheduled.
		err = wait.Poll(time.Second, wait.ForeverTestTimeout, podScheduled(cs, myPod.Namespace, myPod.Name))
		if err != nil {
			t.Errorf("Test %d: failed to schedule a pod: %v", i, err)
		} else {
			t.Logf("Test %d: Pod got scheduled on a schedulable node", i)
		}

		err = cs.Core().Pods(ns.Name).Delete(myPod.Name, metav1.NewDeleteOptions(0))
		if err != nil {
			t.Errorf("Failed to delete pod: %v", err)
		}
		err = cs.Core().Nodes().Delete(schedNode.Name, nil)
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
			- testPodNoAnnotation2 and testPodWithAnnotationFitsDefault2 should NOT be scheduled
	*/
	// 1. create and start default-scheduler
	clientSet := clientset.NewForConfigOrDie(&restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &api.Registry.GroupOrDie(v1.GroupName).GroupVersion}})

	// NOTE: This test cannot run in parallel, because it is creating and deleting
	// non-namespaced objects (Nodes).
	defer clientSet.Core().Nodes().DeleteCollection(nil, metav1.ListOptions{})

	informerFactory := informers.NewSharedInformerFactory(clientSet, 0)
	schedulerConfigFactory := factory.NewConfigFactory(
		v1.DefaultSchedulerName,
		clientSet,
		informerFactory.Core().V1().Nodes(),
		informerFactory.Core().V1().PersistentVolumes(),
		informerFactory.Core().V1().PersistentVolumeClaims(),
		informerFactory.Core().V1().ReplicationControllers(),
		informerFactory.Extensions().V1beta1().ReplicaSets(),
		informerFactory.Apps().V1beta1().StatefulSets(),
		informerFactory.Core().V1().Services(),
		v1.DefaultHardPodAffinitySymmetricWeight,
	)
	schedulerConfig, err := schedulerConfigFactory.Create()
	if err != nil {
		t.Fatalf("Couldn't create scheduler config: %v", err)
	}
	eventBroadcaster := record.NewBroadcaster()
	schedulerConfig.Recorder = eventBroadcaster.NewRecorder(api.Scheme, clientv1.EventSource{Component: v1.DefaultSchedulerName})
	eventBroadcaster.StartRecordingToSink(&clientv1core.EventSinkImpl{Interface: clientv1core.New(clientSet.Core().RESTClient()).Events("")})
	informerFactory.Start(schedulerConfig.StopEverything)
	scheduler.New(schedulerConfig).Run()
	// default-scheduler will be stopped later

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
	clientSet.Core().Nodes().Create(node)

	// 3. create 3 pods for testing
	podWithoutSchedulerName := createPod(clientSet, "pod-without-scheduler-name", "")
	testPod, err := clientSet.Core().Pods(ns.Name).Create(podWithoutSchedulerName)
	if err != nil {
		t.Fatalf("Failed to create pod: %v", err)
	}

	schedulerFitsDefault := "default-scheduler"
	podFitsDefault := createPod(clientSet, "pod-fits-default", schedulerFitsDefault)
	testPodFitsDefault, err := clientSet.Core().Pods(ns.Name).Create(podFitsDefault)
	if err != nil {
		t.Fatalf("Failed to create pod: %v", err)
	}

	schedulerFitsFoo := "foo-scheduler"
	podFitsFoo := createPod(clientSet, "pod-fits-foo", schedulerFitsFoo)
	testPodFitsFoo, err := clientSet.Core().Pods(ns.Name).Create(podFitsFoo)
	if err != nil {
		t.Fatalf("Failed to create pod: %v", err)
	}

	// 4. **check point-1**:
	//		- testPod, testPodFitsDefault should be scheduled
	//		- testPodFitsFoo should NOT be scheduled
	err = wait.Poll(time.Second, time.Second*5, podScheduled(clientSet, testPod.Namespace, testPod.Name))
	if err != nil {
		t.Errorf("Test MultiScheduler: %s Pod not scheduled: %v", testPod.Name, err)
	} else {
		t.Logf("Test MultiScheduler: %s Pod scheduled", testPod.Name)
	}

	err = wait.Poll(time.Second, time.Second*5, podScheduled(clientSet, testPodFitsDefault.Namespace, testPodFitsDefault.Name))
	if err != nil {
		t.Errorf("Test MultiScheduler: %s Pod not scheduled: %v", testPodFitsDefault.Name, err)
	} else {
		t.Logf("Test MultiScheduler: %s Pod scheduled", testPodFitsDefault.Name)
	}

	err = wait.Poll(time.Second, time.Second*5, podScheduled(clientSet, testPodFitsFoo.Namespace, testPodFitsFoo.Name))
	if err == nil {
		t.Errorf("Test MultiScheduler: %s Pod got scheduled, %v", testPodFitsFoo.Name, err)
	} else {
		t.Logf("Test MultiScheduler: %s Pod not scheduled", testPodFitsFoo.Name)
	}

	// 5. create and start a scheduler with name "foo-scheduler"
	clientSet2 := clientset.NewForConfigOrDie(&restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &api.Registry.GroupOrDie(v1.GroupName).GroupVersion}})
	informerFactory2 := informers.NewSharedInformerFactory(clientSet, 0)

	schedulerConfigFactory2 := factory.NewConfigFactory(
		"foo-scheduler",
		clientSet2,
		informerFactory.Core().V1().Nodes(),
		informerFactory.Core().V1().PersistentVolumes(),
		informerFactory.Core().V1().PersistentVolumeClaims(),
		informerFactory.Core().V1().ReplicationControllers(),
		informerFactory.Extensions().V1beta1().ReplicaSets(),
		informerFactory.Apps().V1beta1().StatefulSets(),
		informerFactory.Core().V1().Services(),
		v1.DefaultHardPodAffinitySymmetricWeight,
	)
	schedulerConfig2, err := schedulerConfigFactory2.Create()
	if err != nil {
		t.Errorf("Couldn't create scheduler config: %v", err)
	}
	eventBroadcaster2 := record.NewBroadcaster()
	schedulerConfig2.Recorder = eventBroadcaster2.NewRecorder(api.Scheme, clientv1.EventSource{Component: "foo-scheduler"})
	eventBroadcaster.StartRecordingToSink(&clientv1core.EventSinkImpl{Interface: clientv1core.New(clientSet2.Core().RESTClient()).Events("")})
	informerFactory2.Start(schedulerConfig2.StopEverything)
	scheduler.New(schedulerConfig2).Run()

	defer close(schedulerConfig2.StopEverything)

	//	6. **check point-2**:
	//		- testPodWithAnnotationFitsFoo should be scheduled
	err = wait.Poll(time.Second, time.Second*5, podScheduled(clientSet, testPodFitsFoo.Namespace, testPodFitsFoo.Name))
	if err != nil {
		t.Errorf("Test MultiScheduler: %s Pod not scheduled, %v", testPodFitsFoo.Name, err)
	} else {
		t.Logf("Test MultiScheduler: %s Pod scheduled", testPodFitsFoo.Name)
	}

	//	7. delete the pods that were scheduled by the default scheduler, and stop the default scheduler
	err = clientSet.Core().Pods(ns.Name).Delete(testPod.Name, metav1.NewDeleteOptions(0))
	if err != nil {
		t.Errorf("Failed to delete pod: %v", err)
	}
	err = clientSet.Core().Pods(ns.Name).Delete(testPodFitsDefault.Name, metav1.NewDeleteOptions(0))
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
		testPodNoAnnotation2, err := clientSet.Core().Pods(ns.Name).Create(podWithNoAnnotation2)
		if err != nil {
			t.Fatalf("Failed to create pod: %v", err)
		}
		testPodWithAnnotationFitsDefault2, err := clientSet.Core().Pods(ns.Name).Create(podWithAnnotationFitsDefault2)
		if err != nil {
			t.Fatalf("Failed to create pod: %v", err)
		}

		//	9. **check point-3**:
		//		- testPodNoAnnotation2 and testPodWithAnnotationFitsDefault2 should NOT be scheduled
		err = wait.Poll(time.Second, time.Second*5, podScheduled(clientSet, testPodNoAnnotation2.Namespace, testPodNoAnnotation2.Name))
		if err == nil {
			t.Errorf("Test MultiScheduler: %s Pod got scheduled, %v", testPodNoAnnotation2.Name, err)
		} else {
			t.Logf("Test MultiScheduler: %s Pod not scheduled", testPodNoAnnotation2.Name)
		}
		err = wait.Poll(time.Second, time.Second*5, podScheduled(clientSet, testPodWithAnnotationFitsDefault2.Namespace, testPodWithAnnotationFitsDefault2.Name))
		if err == nil {
			t.Errorf("Test MultiScheduler: %s Pod got scheduled, %v", testPodWithAnnotationFitsDefault2.Name, err)
		} else {
			t.Logf("Test MultiScheduler: %s Pod scheduled", testPodWithAnnotationFitsDefault2.Name)
		}
	*/
}

func createPod(client clientset.Interface, name string, scheduler string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: v1.PodSpec{
			Containers:    []v1.Container{{Name: "container", Image: e2e.GetPauseImageName(client)}},
			SchedulerName: scheduler,
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
	clientSet := clientset.NewForConfigOrDie(&restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &api.Registry.GroupOrDie(v1.GroupName).GroupVersion}})
	informerFactory := informers.NewSharedInformerFactory(clientSet, 0)

	// NOTE: This test cannot run in parallel, because it is creating and deleting
	// non-namespaced objects (Nodes).
	defer clientSet.Core().Nodes().DeleteCollection(nil, metav1.ListOptions{})

	schedulerConfigFactory := factory.NewConfigFactory(
		v1.DefaultSchedulerName,
		clientSet,
		informerFactory.Core().V1().Nodes(),
		informerFactory.Core().V1().PersistentVolumes(),
		informerFactory.Core().V1().PersistentVolumeClaims(),
		informerFactory.Core().V1().ReplicationControllers(),
		informerFactory.Extensions().V1beta1().ReplicaSets(),
		informerFactory.Apps().V1beta1().StatefulSets(),
		informerFactory.Core().V1().Services(),
		v1.DefaultHardPodAffinitySymmetricWeight,
	)
	schedulerConfig, err := schedulerConfigFactory.Create()
	if err != nil {
		t.Fatalf("Couldn't create scheduler config: %v", err)
	}
	eventBroadcaster := record.NewBroadcaster()
	schedulerConfig.Recorder = eventBroadcaster.NewRecorder(api.Scheme, clientv1.EventSource{Component: v1.DefaultSchedulerName})
	eventBroadcaster.StartRecordingToSink(&clientv1core.EventSinkImpl{Interface: clientv1core.New(clientSet.Core().RESTClient()).Events("")})
	informerFactory.Start(schedulerConfig.StopEverything)
	scheduler.New(schedulerConfig).Run()
	// default-scheduler will be stopped later
	defer close(schedulerConfig.StopEverything)

	// 2. create a node without allocatable awareness
	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: "node-allocatable-scheduler-test-node"},
		Spec:       v1.NodeSpec{Unschedulable: false},
		Status: v1.NodeStatus{
			Capacity: v1.ResourceList{
				v1.ResourcePods:   *resource.NewQuantity(32, resource.DecimalSI),
				v1.ResourceCPU:    *resource.NewMilliQuantity(30, resource.DecimalSI),
				v1.ResourceMemory: *resource.NewQuantity(30, resource.BinarySI),
			},
		},
	}

	allocNode, err := clientSet.Core().Nodes().Create(node)
	if err != nil {
		t.Fatalf("Failed to create node: %v", err)
	}

	// 3. create resource pod which requires less than Capacity
	podResource := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "pod-test-allocatable"},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "container",
					Image: e2e.GetPauseImageName(clientSet),
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    *resource.NewMilliQuantity(20, resource.DecimalSI),
							v1.ResourceMemory: *resource.NewQuantity(20, resource.BinarySI),
						},
					},
				},
			},
		},
	}

	testAllocPod, err := clientSet.Core().Pods(ns.Name).Create(podResource)
	if err != nil {
		t.Fatalf("Test allocatable unawareness failed to create pod: %v", err)
	}

	// 4. Test: this test pod should be scheduled since api-server will use Capacity as Allocatable
	err = wait.Poll(time.Second, time.Second*5, podScheduled(clientSet, testAllocPod.Namespace, testAllocPod.Name))
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

	if _, err := clientSet.Core().Nodes().UpdateStatus(allocNode); err != nil {
		t.Fatalf("Failed to update node with Status.Allocatable: %v", err)
	}

	if err := clientSet.Core().Pods(ns.Name).Delete(podResource.Name, &metav1.DeleteOptions{}); err != nil {
		t.Fatalf("Failed to remove first resource pod: %v", err)
	}

	// 6. Make another pod with different name, same resource request
	podResource.ObjectMeta.Name = "pod-test-allocatable2"
	testAllocPod2, err := clientSet.Core().Pods(ns.Name).Create(podResource)
	if err != nil {
		t.Fatalf("Test allocatable awareness failed to create pod: %v", err)
	}

	// 7. Test: this test pod should not be scheduled since it request more than Allocatable
	err = wait.Poll(time.Second, time.Second*5, podScheduled(clientSet, testAllocPod2.Namespace, testAllocPod2.Name))
	if err == nil {
		t.Errorf("Test allocatable awareness: %s Pod got scheduled unexpectedly, %v", testAllocPod2.Name, err)
	} else {
		t.Logf("Test allocatable awareness: %s Pod not scheduled as expected", testAllocPod2.Name)
	}
}
