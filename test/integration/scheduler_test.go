// +build integration,!no-etcd

/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package integration

// This file tests the scheduler.

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apiserver"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/client/record"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/master"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/plugin/pkg/admission/admit"
	"k8s.io/kubernetes/plugin/pkg/scheduler"
	_ "k8s.io/kubernetes/plugin/pkg/scheduler/algorithmprovider"
	"k8s.io/kubernetes/plugin/pkg/scheduler/factory"
	"k8s.io/kubernetes/test/integration/framework"
)

type nodeMutationFunc func(t *testing.T, n *api.Node, nodeStore cache.Store, c *client.Client)

type nodeStateManager struct {
	makeSchedulable   nodeMutationFunc
	makeUnSchedulable nodeMutationFunc
}

func TestUnschedulableNodes(t *testing.T) {
	etcdStorage, err := framework.NewEtcdStorage()
	if err != nil {
		t.Fatalf("Couldn't create etcd storage: %v", err)
	}
	expEtcdStorage, err := framework.NewExtensionsEtcdStorage(nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	storageDestinations := master.NewStorageDestinations()
	storageDestinations.AddAPIGroup("", etcdStorage)
	storageDestinations.AddAPIGroup("extensions", expEtcdStorage)

	storageVersions := make(map[string]string)
	storageVersions[""] = testapi.Default.Version()
	storageVersions["extensions"] = testapi.Extensions.GroupAndVersion()

	framework.DeleteAllEtcdKeys()

	var m *master.Master
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		m.Handler.ServeHTTP(w, req)
	}))
	defer s.Close()

	m = master.New(&master.Config{
		StorageDestinations:   storageDestinations,
		KubeletClient:         client.FakeKubeletClient{},
		EnableCoreControllers: true,
		EnableLogsSupport:     false,
		EnableUISupport:       false,
		EnableIndex:           true,
		APIPrefix:             "/api",
		Authorizer:            apiserver.NewAlwaysAllowAuthorizer(),
		AdmissionControl:      admit.NewAlwaysAdmit(),
		StorageVersions:       storageVersions,
	})

	restClient := client.NewOrDie(&client.Config{Host: s.URL, Version: testapi.Default.Version()})

	schedulerConfigFactory := factory.NewConfigFactory(restClient, nil)
	schedulerConfig, err := schedulerConfigFactory.Create()
	if err != nil {
		t.Fatalf("Couldn't create scheduler config: %v", err)
	}
	eventBroadcaster := record.NewBroadcaster()
	schedulerConfig.Recorder = eventBroadcaster.NewRecorder(api.EventSource{Component: "scheduler"})
	eventBroadcaster.StartRecordingToSink(restClient.Events(""))
	scheduler.New(schedulerConfig).Run()

	defer close(schedulerConfig.StopEverything)

	DoTestUnschedulableNodes(t, restClient, schedulerConfigFactory.NodeLister.Store)
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
func waitForReflection(s cache.Store, key string, passFunc func(n interface{}) bool) error {
	return wait.Poll(time.Millisecond*10, time.Second*20, func() (bool, error) {
		if n, _, err := s.GetByKey(key); err == nil && passFunc(n) {
			return true, nil
		}
		return false, nil
	})
}

func DoTestUnschedulableNodes(t *testing.T, restClient *client.Client, nodeStore cache.Store) {
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
				err = waitForReflection(s, nodeKey, func(node interface{}) bool {
					// An unschedulable node should get deleted from the store
					return node == nil
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
				err = waitForReflection(s, nodeKey, func(node interface{}) bool {
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
				err = waitForReflection(s, nodeKey, func(node interface{}) bool {
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
				waitForReflection(s, nodeKey, func(node interface{}) bool {
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
				Containers: []api.Container{{Name: "container", Image: "kubernetes/pause:go"}},
			},
		}
		myPod, err := restClient.Pods(api.NamespaceDefault).Create(pod)
		if err != nil {
			t.Fatalf("Failed to create pod: %v", err)
		}

		// There are no schedulable nodes - the pod shouldn't be scheduled.
		err = wait.Poll(time.Second, util.ForeverTestTimeout, podScheduled(restClient, myPod.Namespace, myPod.Name))
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
		err = wait.Poll(time.Second, util.ForeverTestTimeout, podScheduled(restClient, myPod.Namespace, myPod.Name))
		if err != nil {
			t.Errorf("Test %d: failed to schedule a pod: %v", i, err)
		} else {
			t.Logf("Test %d: Pod got scheduled on a schedulable node", i)
		}

		err = restClient.Pods(api.NamespaceDefault).Delete(myPod.Name, api.NewDeleteOptions(0))
		if err != nil {
			t.Errorf("Failed to delete pod: %v", err)
		}
		err = restClient.Nodes().Delete(schedNode.Name)
		if err != nil {
			t.Errorf("Failed to delete node: %v", err)
		}
	}
}

func BenchmarkScheduling(b *testing.B) {
	etcdStorage, err := framework.NewEtcdStorage()
	if err != nil {
		b.Fatalf("Couldn't create etcd storage: %v", err)
	}
	expEtcdStorage, err := framework.NewExtensionsEtcdStorage(nil)
	if err != nil {
		b.Fatalf("unexpected error: %v", err)
	}

	storageDestinations := master.NewStorageDestinations()
	storageDestinations.AddAPIGroup("", etcdStorage)
	storageDestinations.AddAPIGroup("extensions", expEtcdStorage)

	storageVersions := make(map[string]string)
	storageVersions[""] = testapi.Default.Version()
	storageVersions["extensions"] = testapi.Extensions.GroupAndVersion()

	framework.DeleteAllEtcdKeys()

	var m *master.Master
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		m.Handler.ServeHTTP(w, req)
	}))
	defer s.Close()

	m = master.New(&master.Config{
		StorageDestinations:   storageDestinations,
		KubeletClient:         client.FakeKubeletClient{},
		EnableCoreControllers: true,
		EnableLogsSupport:     false,
		EnableUISupport:       false,
		EnableIndex:           true,
		APIPrefix:             "/api",
		Authorizer:            apiserver.NewAlwaysAllowAuthorizer(),
		AdmissionControl:      admit.NewAlwaysAdmit(),
		StorageVersions:       storageVersions,
	})

	c := client.NewOrDie(&client.Config{
		Host:    s.URL,
		Version: testapi.Default.Version(),
		QPS:     5000.0,
		Burst:   5000,
	})

	schedulerConfigFactory := factory.NewConfigFactory(c, nil)
	schedulerConfig, err := schedulerConfigFactory.Create()
	if err != nil {
		b.Fatalf("Couldn't create scheduler config: %v", err)
	}
	eventBroadcaster := record.NewBroadcaster()
	schedulerConfig.Recorder = eventBroadcaster.NewRecorder(api.EventSource{Component: "scheduler"})
	eventBroadcaster.StartRecordingToSink(c.Events(""))
	scheduler.New(schedulerConfig).Run()

	defer close(schedulerConfig.StopEverything)

	makeNNodes(c, 1000)
	N := b.N
	b.ResetTimer()
	makeNPods(c, N)
	for {
		objs := schedulerConfigFactory.ScheduledPodLister.Store.List()
		if len(objs) >= N {
			fmt.Printf("%v pods scheduled.\n", len(objs))
			/* // To prove that this actually works:
			for _, o := range objs {
				fmt.Printf("%s\n", o.(*api.Pod).Spec.NodeName)
			}
			*/
			break
		}
		time.Sleep(time.Millisecond)
	}
	b.StopTimer()
}

func makeNNodes(c client.Interface, N int) {
	baseNode := &api.Node{
		ObjectMeta: api.ObjectMeta{
			GenerateName: "scheduler-test-node-",
		},
		Spec: api.NodeSpec{
			ExternalID: "foobar",
		},
		Status: api.NodeStatus{
			Capacity: api.ResourceList{
				api.ResourcePods:   *resource.NewQuantity(32, resource.DecimalSI),
				api.ResourceCPU:    resource.MustParse("4"),
				api.ResourceMemory: resource.MustParse("32Gi"),
			},
			Phase: api.NodeRunning,
			Conditions: []api.NodeCondition{
				{Type: api.NodeReady, Status: api.ConditionTrue},
			},
		},
	}
	for i := 0; i < N; i++ {
		if _, err := c.Nodes().Create(baseNode); err != nil {
			panic("error creating node: " + err.Error())
		}
	}
}

func makeNPods(c client.Interface, N int) {
	basePod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			GenerateName: "scheduler-test-pod-",
		},
		Spec: api.PodSpec{
			Containers: []api.Container{{
				Name:  "pause",
				Image: "gcr.io/google_containers/pause:1.0",
				Resources: api.ResourceRequirements{
					Limits: api.ResourceList{
						api.ResourceCPU:    resource.MustParse("100m"),
						api.ResourceMemory: resource.MustParse("500Mi"),
					},
					Requests: api.ResourceList{
						api.ResourceCPU:    resource.MustParse("100m"),
						api.ResourceMemory: resource.MustParse("500Mi"),
					},
				},
			}},
		},
	}
	wg := sync.WaitGroup{}
	threads := 30
	wg.Add(threads)
	remaining := make(chan int, N)
	go func() {
		for i := 0; i < N; i++ {
			remaining <- i
		}
		close(remaining)
	}()
	for i := 0; i < threads; i++ {
		go func() {
			defer wg.Done()
			for {
				_, ok := <-remaining
				if !ok {
					return
				}
				for {
					_, err := c.Pods("default").Create(basePod)
					if err == nil {
						break
					}
				}
			}
		}()
	}
	wg.Wait()
}
