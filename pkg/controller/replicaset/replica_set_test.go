/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

// If you make changes to this file, you should also make the corresponding change in ReplicationController.

package replicaset

import (
	"fmt"
	"math/rand"
	"net/http/httptest"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/cache"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	"k8s.io/kubernetes/pkg/client/testing/core"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/securitycontext"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/sets"
	utiltesting "k8s.io/kubernetes/pkg/util/testing"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/watch"
)

var alwaysReady = func() bool { return true }

func getKey(rs *extensions.ReplicaSet, t *testing.T) string {
	if key, err := controller.KeyFunc(rs); err != nil {
		t.Errorf("Unexpected error getting key for ReplicaSet %v: %v", rs.Name, err)
		return ""
	} else {
		return key
	}
}

func newReplicaSet(replicas int, selectorMap map[string]string) *extensions.ReplicaSet {
	rs := &extensions.ReplicaSet{
		TypeMeta: unversioned.TypeMeta{APIVersion: testapi.Default.GroupVersion().String()},
		ObjectMeta: api.ObjectMeta{
			UID:             util.NewUUID(),
			Name:            "foobar",
			Namespace:       api.NamespaceDefault,
			ResourceVersion: "18",
		},
		Spec: extensions.ReplicaSetSpec{
			Replicas: replicas,
			Selector: &unversioned.LabelSelector{MatchLabels: selectorMap},
			Template: &api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: map[string]string{
						"name": "foo",
						"type": "production",
					},
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Image: "foo/bar",
							TerminationMessagePath: api.TerminationMessagePathDefault,
							ImagePullPolicy:        api.PullIfNotPresent,
							SecurityContext:        securitycontext.ValidSecurityContextWithContainerDefaults(),
						},
					},
					RestartPolicy: api.RestartPolicyAlways,
					DNSPolicy:     api.DNSDefault,
					NodeSelector: map[string]string{
						"baz": "blah",
					},
				},
			},
		},
	}
	return rs
}

// create count pods with the given phase for the given ReplicaSet (same selectors and namespace), and add them to the store.
func newPodList(store cache.Store, count int, status api.PodPhase, labelMap map[string]string, rs *extensions.ReplicaSet) *api.PodList {
	pods := []api.Pod{}
	for i := 0; i < count; i++ {
		newPod := api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name:      fmt.Sprintf("pod%d", i),
				Labels:    labelMap,
				Namespace: rs.Namespace,
			},
			Status: api.PodStatus{Phase: status},
		}
		if store != nil {
			store.Add(&newPod)
		}
		pods = append(pods, newPod)
	}
	return &api.PodList{
		Items: pods,
	}
}

func validateSyncReplicaSet(t *testing.T, fakePodControl *controller.FakePodControl, expectedCreates, expectedDeletes int) {
	if len(fakePodControl.Templates) != expectedCreates {
		t.Errorf("Unexpected number of creates.  Expected %d, saw %d\n", expectedCreates, len(fakePodControl.Templates))
	}
	if len(fakePodControl.DeletePodName) != expectedDeletes {
		t.Errorf("Unexpected number of deletes.  Expected %d, saw %d\n", expectedDeletes, len(fakePodControl.DeletePodName))
	}
}

func replicaSetResourceName() string {
	return "replicasets"
}

type serverResponse struct {
	statusCode int
	obj        interface{}
}

func TestSyncReplicaSetDoesNothing(t *testing.T) {
	client := clientset.NewForConfigOrDie(&client.Config{Host: "", ContentConfig: client.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}})
	fakePodControl := controller.FakePodControl{}
	manager := NewReplicaSetController(client, controller.NoResyncPeriodFunc, BurstReplicas, 0)
	manager.podStoreSynced = alwaysReady

	// 2 running pods, a controller with 2 replicas, sync is a no-op
	labelMap := map[string]string{"foo": "bar"}
	rsSpec := newReplicaSet(2, labelMap)
	manager.rsStore.Store.Add(rsSpec)
	newPodList(manager.podStore.Store, 2, api.PodRunning, labelMap, rsSpec)

	manager.podControl = &fakePodControl
	manager.syncReplicaSet(getKey(rsSpec, t))
	validateSyncReplicaSet(t, &fakePodControl, 0, 0)
}

func TestSyncReplicaSetDeletes(t *testing.T) {
	client := clientset.NewForConfigOrDie(&client.Config{Host: "", ContentConfig: client.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}})
	fakePodControl := controller.FakePodControl{}
	manager := NewReplicaSetController(client, controller.NoResyncPeriodFunc, BurstReplicas, 0)
	manager.podStoreSynced = alwaysReady
	manager.podControl = &fakePodControl

	// 2 running pods and a controller with 1 replica, one pod delete expected
	labelMap := map[string]string{"foo": "bar"}
	rsSpec := newReplicaSet(1, labelMap)
	manager.rsStore.Store.Add(rsSpec)
	newPodList(manager.podStore.Store, 2, api.PodRunning, labelMap, rsSpec)

	manager.syncReplicaSet(getKey(rsSpec, t))
	validateSyncReplicaSet(t, &fakePodControl, 0, 1)
}

func TestDeleteFinalStateUnknown(t *testing.T) {
	client := clientset.NewForConfigOrDie(&client.Config{Host: "", ContentConfig: client.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}})
	fakePodControl := controller.FakePodControl{}
	manager := NewReplicaSetController(client, controller.NoResyncPeriodFunc, BurstReplicas, 0)
	manager.podStoreSynced = alwaysReady
	manager.podControl = &fakePodControl

	received := make(chan string)
	manager.syncHandler = func(key string) error {
		received <- key
		return nil
	}

	// The DeletedFinalStateUnknown object should cause the ReplicaSet manager to insert
	// the controller matching the selectors of the deleted pod into the work queue.
	labelMap := map[string]string{"foo": "bar"}
	rsSpec := newReplicaSet(1, labelMap)
	manager.rsStore.Store.Add(rsSpec)
	pods := newPodList(nil, 1, api.PodRunning, labelMap, rsSpec)
	manager.deletePod(cache.DeletedFinalStateUnknown{Key: "foo", Obj: &pods.Items[0]})

	go manager.worker()

	expected := getKey(rsSpec, t)
	select {
	case key := <-received:
		if key != expected {
			t.Errorf("Unexpected sync all for ReplicaSet %v, expected %v", key, expected)
		}
	case <-time.After(wait.ForeverTestTimeout):
		t.Errorf("Processing DeleteFinalStateUnknown took longer than expected")
	}
}

func TestSyncReplicaSetCreates(t *testing.T) {
	client := clientset.NewForConfigOrDie(&client.Config{Host: "", ContentConfig: client.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}})
	manager := NewReplicaSetController(client, controller.NoResyncPeriodFunc, BurstReplicas, 0)
	manager.podStoreSynced = alwaysReady

	// A controller with 2 replicas and no pods in the store, 2 creates expected
	labelMap := map[string]string{"foo": "bar"}
	rs := newReplicaSet(2, labelMap)
	manager.rsStore.Store.Add(rs)

	fakePodControl := controller.FakePodControl{}
	manager.podControl = &fakePodControl
	manager.syncReplicaSet(getKey(rs, t))
	validateSyncReplicaSet(t, &fakePodControl, 2, 0)
}

func TestStatusUpdatesWithoutReplicasChange(t *testing.T) {
	// Setup a fake server to listen for requests, and run the ReplicaSet controller in steady state
	fakeHandler := utiltesting.FakeHandler{
		StatusCode:   200,
		ResponseBody: "",
	}
	testServer := httptest.NewServer(&fakeHandler)
	defer testServer.Close()
	client := clientset.NewForConfigOrDie(&client.Config{Host: testServer.URL, ContentConfig: client.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}})
	manager := NewReplicaSetController(client, controller.NoResyncPeriodFunc, BurstReplicas, 0)
	manager.podStoreSynced = alwaysReady

	// Steady state for the ReplicaSet, no Status.Replicas updates expected
	activePods := 5
	labelMap := map[string]string{"foo": "bar"}
	rs := newReplicaSet(activePods, labelMap)
	manager.rsStore.Store.Add(rs)
	rs.Status = extensions.ReplicaSetStatus{Replicas: activePods}
	newPodList(manager.podStore.Store, activePods, api.PodRunning, labelMap, rs)

	fakePodControl := controller.FakePodControl{}
	manager.podControl = &fakePodControl
	manager.syncReplicaSet(getKey(rs, t))

	validateSyncReplicaSet(t, &fakePodControl, 0, 0)
	if fakeHandler.RequestReceived != nil {
		t.Errorf("Unexpected update when pods and ReplicaSets are in a steady state")
	}

	// This response body is just so we don't err out decoding the http response, all
	// we care about is the request body sent below.
	response := runtime.EncodeOrDie(testapi.Extensions.Codec(), &extensions.ReplicaSet{})
	fakeHandler.ResponseBody = response

	rs.Generation = rs.Generation + 1
	manager.syncReplicaSet(getKey(rs, t))

	rs.Status.ObservedGeneration = rs.Generation
	updatedRc := runtime.EncodeOrDie(testapi.Extensions.Codec(), rs)
	fakeHandler.ValidateRequest(t, testapi.Extensions.ResourcePath(replicaSetResourceName(), rs.Namespace, rs.Name)+"/status", "PUT", &updatedRc)
}

func TestControllerUpdateReplicas(t *testing.T) {
	// This is a happy server just to record the PUT request we expect for status.Replicas
	fakeHandler := utiltesting.FakeHandler{
		StatusCode:   200,
		ResponseBody: "",
	}
	testServer := httptest.NewServer(&fakeHandler)
	defer testServer.Close()

	client := clientset.NewForConfigOrDie(&client.Config{Host: testServer.URL, ContentConfig: client.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}})
	manager := NewReplicaSetController(client, controller.NoResyncPeriodFunc, BurstReplicas, 0)
	manager.podStoreSynced = alwaysReady

	// Insufficient number of pods in the system, and Status.Replicas is wrong;
	// Status.Replica should update to match number of pods in system, 1 new pod should be created.
	labelMap := map[string]string{"foo": "bar"}
	rs := newReplicaSet(5, labelMap)
	manager.rsStore.Store.Add(rs)
	rs.Status = extensions.ReplicaSetStatus{Replicas: 2, ObservedGeneration: 0}
	rs.Generation = 1
	newPodList(manager.podStore.Store, 4, api.PodRunning, labelMap, rs)

	// This response body is just so we don't err out decoding the http response
	response := runtime.EncodeOrDie(testapi.Extensions.Codec(), &extensions.ReplicaSet{})
	fakeHandler.ResponseBody = response

	fakePodControl := controller.FakePodControl{}
	manager.podControl = &fakePodControl

	manager.syncReplicaSet(getKey(rs, t))

	// 1. Status.Replicas should go up from 2->4 even though we created 5-4=1 pod.
	// 2. Every update to the status should include the Generation of the spec.
	rs.Status = extensions.ReplicaSetStatus{Replicas: 4, ObservedGeneration: 1}

	decRc := runtime.EncodeOrDie(testapi.Extensions.Codec(), rs)
	fakeHandler.ValidateRequest(t, testapi.Extensions.ResourcePath(replicaSetResourceName(), rs.Namespace, rs.Name)+"/status", "PUT", &decRc)
	validateSyncReplicaSet(t, &fakePodControl, 1, 0)
}

func TestSyncReplicaSetDormancy(t *testing.T) {
	// Setup a test server so we can lie about the current state of pods
	fakeHandler := utiltesting.FakeHandler{
		StatusCode:   200,
		ResponseBody: "",
	}
	testServer := httptest.NewServer(&fakeHandler)
	defer testServer.Close()
	client := clientset.NewForConfigOrDie(&client.Config{Host: testServer.URL, ContentConfig: client.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}})

	fakePodControl := controller.FakePodControl{}
	manager := NewReplicaSetController(client, controller.NoResyncPeriodFunc, BurstReplicas, 0)
	manager.podStoreSynced = alwaysReady
	manager.podControl = &fakePodControl

	labelMap := map[string]string{"foo": "bar"}
	rsSpec := newReplicaSet(2, labelMap)
	manager.rsStore.Store.Add(rsSpec)
	newPodList(manager.podStore.Store, 1, api.PodRunning, labelMap, rsSpec)

	// Creates a replica and sets expectations
	rsSpec.Status.Replicas = 1
	manager.syncReplicaSet(getKey(rsSpec, t))
	validateSyncReplicaSet(t, &fakePodControl, 1, 0)

	// Expectations prevents replicas but not an update on status
	rsSpec.Status.Replicas = 0
	fakePodControl.Clear()
	manager.syncReplicaSet(getKey(rsSpec, t))
	validateSyncReplicaSet(t, &fakePodControl, 0, 0)

	// Get the key for the controller
	rsKey, err := controller.KeyFunc(rsSpec)
	if err != nil {
		t.Errorf("Couldn't get key for object %+v: %v", rsSpec, err)
	}

	// Lowering expectations should lead to a sync that creates a replica, however the
	// fakePodControl error will prevent this, leaving expectations at 0, 0
	manager.expectations.CreationObserved(rsKey)
	rsSpec.Status.Replicas = 1
	fakePodControl.Clear()
	fakePodControl.Err = fmt.Errorf("Fake Error")

	manager.syncReplicaSet(getKey(rsSpec, t))
	validateSyncReplicaSet(t, &fakePodControl, 0, 0)

	// This replica should not need a Lowering of expectations, since the previous create failed
	fakePodControl.Err = nil
	manager.syncReplicaSet(getKey(rsSpec, t))
	validateSyncReplicaSet(t, &fakePodControl, 1, 0)

	// 1 PUT for the ReplicaSet status during dormancy window.
	// Note that the pod creates go through pod control so they're not recorded.
	fakeHandler.ValidateRequestCount(t, 1)
}

func TestPodControllerLookup(t *testing.T) {
	manager := NewReplicaSetController(clientset.NewForConfigOrDie(&client.Config{Host: "", ContentConfig: client.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}}), controller.NoResyncPeriodFunc, BurstReplicas, 0)
	manager.podStoreSynced = alwaysReady
	testCases := []struct {
		inRSs     []*extensions.ReplicaSet
		pod       *api.Pod
		outRSName string
	}{
		// pods without labels don't match any ReplicaSets
		{
			inRSs: []*extensions.ReplicaSet{
				{ObjectMeta: api.ObjectMeta{Name: "basic"}}},
			pod:       &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo1", Namespace: api.NamespaceAll}},
			outRSName: "",
		},
		// Matching labels, not namespace
		{
			inRSs: []*extensions.ReplicaSet{
				{
					ObjectMeta: api.ObjectMeta{Name: "foo"},
					Spec: extensions.ReplicaSetSpec{
						Selector: &unversioned.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}},
					},
				},
			},
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "foo2", Namespace: "ns", Labels: map[string]string{"foo": "bar"}}},
			outRSName: "",
		},
		// Matching ns and labels returns the key to the ReplicaSet, not the ReplicaSet name
		{
			inRSs: []*extensions.ReplicaSet{
				{
					ObjectMeta: api.ObjectMeta{Name: "bar", Namespace: "ns"},
					Spec: extensions.ReplicaSetSpec{
						Selector: &unversioned.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}},
					},
				},
			},
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "foo3", Namespace: "ns", Labels: map[string]string{"foo": "bar"}}},
			outRSName: "bar",
		},
	}
	for _, c := range testCases {
		for _, r := range c.inRSs {
			manager.rsStore.Add(r)
		}
		if rs := manager.getPodReplicaSet(c.pod); rs != nil {
			if c.outRSName != rs.Name {
				t.Errorf("Got replica set %+v expected %+v", rs.Name, c.outRSName)
			}
		} else if c.outRSName != "" {
			t.Errorf("Expected a replica set %v pod %v, found none", c.outRSName, c.pod.Name)
		}
	}
}

type FakeWatcher struct {
	w *watch.FakeWatcher
	*fake.Clientset
}

func TestWatchControllers(t *testing.T) {
	fakeWatch := watch.NewFake()
	client := &fake.Clientset{}
	client.AddWatchReactor("*", core.DefaultWatchReactor(fakeWatch, nil))
	manager := NewReplicaSetController(client, controller.NoResyncPeriodFunc, BurstReplicas, 0)
	manager.podStoreSynced = alwaysReady

	var testRSSpec extensions.ReplicaSet
	received := make(chan string)

	// The update sent through the fakeWatcher should make its way into the workqueue,
	// and eventually into the syncHandler. The handler validates the received controller
	// and closes the received channel to indicate that the test can finish.
	manager.syncHandler = func(key string) error {

		obj, exists, err := manager.rsStore.Store.GetByKey(key)
		if !exists || err != nil {
			t.Errorf("Expected to find replica set under key %v", key)
		}
		rsSpec := *obj.(*extensions.ReplicaSet)
		if !api.Semantic.DeepDerivative(rsSpec, testRSSpec) {
			t.Errorf("Expected %#v, but got %#v", testRSSpec, rsSpec)
		}
		close(received)
		return nil
	}
	// Start only the ReplicaSet watcher and the workqueue, send a watch event,
	// and make sure it hits the sync method.
	stopCh := make(chan struct{})
	defer close(stopCh)
	go manager.rsController.Run(stopCh)
	go wait.Until(manager.worker, 10*time.Millisecond, stopCh)

	testRSSpec.Name = "foo"
	fakeWatch.Add(&testRSSpec)

	select {
	case <-received:
	case <-time.After(wait.ForeverTestTimeout):
		t.Errorf("Expected 1 call but got 0")
	}
}

func TestWatchPods(t *testing.T) {
	fakeWatch := watch.NewFake()
	client := &fake.Clientset{}
	client.AddWatchReactor("*", core.DefaultWatchReactor(fakeWatch, nil))
	manager := NewReplicaSetController(client, controller.NoResyncPeriodFunc, BurstReplicas, 0)
	manager.podStoreSynced = alwaysReady

	// Put one ReplicaSet and one pod into the controller's stores
	labelMap := map[string]string{"foo": "bar"}
	testRSSpec := newReplicaSet(1, labelMap)
	manager.rsStore.Store.Add(testRSSpec)
	received := make(chan string)
	// The pod update sent through the fakeWatcher should figure out the managing ReplicaSet and
	// send it into the syncHandler.
	manager.syncHandler = func(key string) error {

		obj, exists, err := manager.rsStore.Store.GetByKey(key)
		if !exists || err != nil {
			t.Errorf("Expected to find replica set under key %v", key)
		}
		rsSpec := obj.(*extensions.ReplicaSet)
		if !api.Semantic.DeepDerivative(rsSpec, testRSSpec) {
			t.Errorf("\nExpected %#v,\nbut got %#v", testRSSpec, rsSpec)
		}
		close(received)
		return nil
	}
	// Start only the pod watcher and the workqueue, send a watch event,
	// and make sure it hits the sync method for the right ReplicaSet.
	stopCh := make(chan struct{})
	defer close(stopCh)
	go manager.podController.Run(stopCh)
	go wait.Until(manager.worker, 10*time.Millisecond, stopCh)

	pods := newPodList(nil, 1, api.PodRunning, labelMap, testRSSpec)
	testPod := pods.Items[0]
	testPod.Status.Phase = api.PodFailed
	fakeWatch.Add(&testPod)

	select {
	case <-received:
	case <-time.After(wait.ForeverTestTimeout):
		t.Errorf("Expected 1 call but got 0")
	}
}

func TestUpdatePods(t *testing.T) {
	manager := NewReplicaSetController(fake.NewSimpleClientset(), controller.NoResyncPeriodFunc, BurstReplicas, 0)
	manager.podStoreSynced = alwaysReady

	received := make(chan string)

	manager.syncHandler = func(key string) error {
		obj, exists, err := manager.rsStore.Store.GetByKey(key)
		if !exists || err != nil {
			t.Errorf("Expected to find replica set under key %v", key)
		}
		received <- obj.(*extensions.ReplicaSet).Name
		return nil
	}

	stopCh := make(chan struct{})
	defer close(stopCh)
	go wait.Until(manager.worker, 10*time.Millisecond, stopCh)

	// Put 2 ReplicaSets and one pod into the controller's stores
	labelMap1 := map[string]string{"foo": "bar"}
	testRSSpec1 := newReplicaSet(1, labelMap1)
	manager.rsStore.Store.Add(testRSSpec1)
	testRSSpec2 := *testRSSpec1
	labelMap2 := map[string]string{"bar": "foo"}
	testRSSpec2.Spec.Selector = &unversioned.LabelSelector{MatchLabels: labelMap2}
	testRSSpec2.Name = "barfoo"
	manager.rsStore.Store.Add(&testRSSpec2)

	// Put one pod in the podStore
	pod1 := newPodList(manager.podStore.Store, 1, api.PodRunning, labelMap1, testRSSpec1).Items[0]
	pod2 := pod1
	pod2.Labels = labelMap2

	// Send an update of the same pod with modified labels, and confirm we get a sync request for
	// both controllers
	manager.updatePod(&pod1, &pod2)

	expected := sets.NewString(testRSSpec1.Name, testRSSpec2.Name)
	for _, name := range expected.List() {
		t.Logf("Expecting update for %+v", name)
		select {
		case got := <-received:
			if !expected.Has(got) {
				t.Errorf("Expected keys %#v got %v", expected, got)
			}
		case <-time.After(wait.ForeverTestTimeout):
			t.Errorf("Expected update notifications for replica sets within 100ms each")
		}
	}
}

func TestControllerUpdateRequeue(t *testing.T) {
	// This server should force a requeue of the controller because it fails to update status.Replicas.
	fakeHandler := utiltesting.FakeHandler{
		StatusCode:   500,
		ResponseBody: "",
	}
	testServer := httptest.NewServer(&fakeHandler)
	defer testServer.Close()

	client := clientset.NewForConfigOrDie(&client.Config{Host: testServer.URL, ContentConfig: client.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}})
	manager := NewReplicaSetController(client, controller.NoResyncPeriodFunc, BurstReplicas, 0)
	manager.podStoreSynced = alwaysReady

	labelMap := map[string]string{"foo": "bar"}
	rs := newReplicaSet(1, labelMap)
	manager.rsStore.Store.Add(rs)
	rs.Status = extensions.ReplicaSetStatus{Replicas: 2}
	newPodList(manager.podStore.Store, 1, api.PodRunning, labelMap, rs)

	fakePodControl := controller.FakePodControl{}
	manager.podControl = &fakePodControl

	manager.syncReplicaSet(getKey(rs, t))

	ch := make(chan interface{})
	go func() {
		item, _ := manager.queue.Get()
		ch <- item
	}()
	select {
	case key := <-ch:
		expectedKey := getKey(rs, t)
		if key != expectedKey {
			t.Errorf("Expected requeue of replica set with key %s got %s", expectedKey, key)
		}
	case <-time.After(wait.ForeverTestTimeout):
		manager.queue.ShutDown()
		t.Errorf("Expected to find a ReplicaSet in the queue, found none.")
	}
	// 1 Update and 1 GET, both of which fail
	fakeHandler.ValidateRequestCount(t, 2)
}

func TestControllerUpdateStatusWithFailure(t *testing.T) {
	rs := newReplicaSet(1, map[string]string{"foo": "bar"})
	fakeClient := &fake.Clientset{}
	fakeClient.AddReactor("get", "replicasets", func(action core.Action) (bool, runtime.Object, error) { return true, rs, nil })
	fakeClient.AddReactor("*", "*", func(action core.Action) (bool, runtime.Object, error) {
		return true, &extensions.ReplicaSet{}, fmt.Errorf("Fake error")
	})
	fakeRSClient := fakeClient.Extensions().ReplicaSets("default")
	numReplicas := 10
	updateReplicaCount(fakeRSClient, *rs, numReplicas)
	updates, gets := 0, 0
	for _, a := range fakeClient.Actions() {
		if a.GetResource() != "replicasets" {
			t.Errorf("Unexpected action %+v", a)
			continue
		}

		switch action := a.(type) {
		case testclient.GetAction:
			gets++
			// Make sure the get is for the right ReplicaSet even though the update failed.
			if action.GetName() != rs.Name {
				t.Errorf("Expected get for ReplicaSet %v, got %+v instead", rs.Name, action.GetName())
			}
		case testclient.UpdateAction:
			updates++
			// Confirm that the update has the right status.Replicas even though the Get
			// returned a ReplicaSet with replicas=1.
			if c, ok := action.GetObject().(*extensions.ReplicaSet); !ok {
				t.Errorf("Expected a ReplicaSet as the argument to update, got %T", c)
			} else if c.Status.Replicas != numReplicas {
				t.Errorf("Expected update for ReplicaSet to contain replicas %v, got %v instead",
					numReplicas, c.Status.Replicas)
			}
		default:
			t.Errorf("Unexpected action %+v", a)
			break
		}
	}
	if gets != 1 || updates != 2 {
		t.Errorf("Expected 1 get and 2 updates, got %d gets %d updates", gets, updates)
	}
}

func doTestControllerBurstReplicas(t *testing.T, burstReplicas, numReplicas int) {
	client := clientset.NewForConfigOrDie(&client.Config{Host: "", ContentConfig: client.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}})
	fakePodControl := controller.FakePodControl{}
	manager := NewReplicaSetController(client, controller.NoResyncPeriodFunc, burstReplicas, 0)
	manager.podStoreSynced = alwaysReady
	manager.podControl = &fakePodControl

	labelMap := map[string]string{"foo": "bar"}
	rsSpec := newReplicaSet(numReplicas, labelMap)
	manager.rsStore.Store.Add(rsSpec)

	expectedPods := 0
	pods := newPodList(nil, numReplicas, api.PodPending, labelMap, rsSpec)

	rsKey, err := controller.KeyFunc(rsSpec)
	if err != nil {
		t.Errorf("Couldn't get key for object %+v: %v", rsSpec, err)
	}

	// Size up the controller, then size it down, and confirm the expected create/delete pattern
	for _, replicas := range []int{numReplicas, 0} {

		rsSpec.Spec.Replicas = replicas
		manager.rsStore.Store.Add(rsSpec)

		for i := 0; i < numReplicas; i += burstReplicas {
			manager.syncReplicaSet(getKey(rsSpec, t))

			// The store accrues active pods. It's also used by the ReplicaSet to determine how many
			// replicas to create.
			activePods := len(manager.podStore.Store.List())
			if replicas != 0 {
				// This is the number of pods currently "in flight". They were created by the
				// ReplicaSet controller above, which then puts the ReplicaSet to sleep till
				// all of them have been observed.
				expectedPods = replicas - activePods
				if expectedPods > burstReplicas {
					expectedPods = burstReplicas
				}
				// This validates the ReplicaSet manager sync actually created pods
				validateSyncReplicaSet(t, &fakePodControl, expectedPods, 0)

				// This simulates the watch events for all but 1 of the expected pods.
				// None of these should wake the controller because it has expectations==BurstReplicas.
				for i := 0; i < expectedPods-1; i++ {
					manager.podStore.Store.Add(&pods.Items[i])
					manager.addPod(&pods.Items[i])
				}

				podExp, exists, err := manager.expectations.GetExpectations(rsKey)
				if !exists || err != nil {
					t.Fatalf("Did not find expectations for ReplicaSet.")
				}
				if add, _ := podExp.GetExpectations(); add != 1 {
					t.Fatalf("Expectations are wrong %v", podExp)
				}
			} else {
				expectedPods = (replicas - activePods) * -1
				if expectedPods > burstReplicas {
					expectedPods = burstReplicas
				}
				validateSyncReplicaSet(t, &fakePodControl, 0, expectedPods)
				for i := 0; i < expectedPods-1; i++ {
					manager.podStore.Store.Delete(&pods.Items[i])
					manager.deletePod(&pods.Items[i])
				}
				podExp, exists, err := manager.expectations.GetExpectations(rsKey)
				if !exists || err != nil {
					t.Fatalf("Did not find expectations for ReplicaSet.")
				}
				if _, del := podExp.GetExpectations(); del != 1 {
					t.Fatalf("Expectations are wrong %v", podExp)
				}
			}

			// Check that the ReplicaSet didn't take any action for all the above pods
			fakePodControl.Clear()
			manager.syncReplicaSet(getKey(rsSpec, t))
			validateSyncReplicaSet(t, &fakePodControl, 0, 0)

			// Create/Delete the last pod
			// The last add pod will decrease the expectation of the ReplicaSet to 0,
			// which will cause it to create/delete the remaining replicas up to burstReplicas.
			if replicas != 0 {
				manager.podStore.Store.Add(&pods.Items[expectedPods-1])
				manager.addPod(&pods.Items[expectedPods-1])
			} else {
				manager.podStore.Store.Delete(&pods.Items[expectedPods-1])
				manager.deletePod(&pods.Items[expectedPods-1])
			}
			pods.Items = pods.Items[expectedPods:]
		}

		// Confirm that we've created the right number of replicas
		activePods := len(manager.podStore.Store.List())
		if activePods != rsSpec.Spec.Replicas {
			t.Fatalf("Unexpected number of active pods, expected %d, got %d", rsSpec.Spec.Replicas, activePods)
		}
		// Replenish the pod list, since we cut it down sizing up
		pods = newPodList(nil, replicas, api.PodRunning, labelMap, rsSpec)
	}
}

func TestControllerBurstReplicas(t *testing.T) {
	doTestControllerBurstReplicas(t, 5, 30)
	doTestControllerBurstReplicas(t, 5, 12)
	doTestControllerBurstReplicas(t, 3, 2)
}

type FakeRSExpectations struct {
	*controller.ControllerExpectations
	satisfied    bool
	expSatisfied func()
}

func (fe FakeRSExpectations) SatisfiedExpectations(controllerKey string) bool {
	fe.expSatisfied()
	return fe.satisfied
}

// TestRSSyncExpectations tests that a pod cannot sneak in between counting active pods
// and checking expectations.
func TestRSSyncExpectations(t *testing.T) {
	client := clientset.NewForConfigOrDie(&client.Config{Host: "", ContentConfig: client.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}})
	fakePodControl := controller.FakePodControl{}
	manager := NewReplicaSetController(client, controller.NoResyncPeriodFunc, 2, 0)
	manager.podStoreSynced = alwaysReady
	manager.podControl = &fakePodControl

	labelMap := map[string]string{"foo": "bar"}
	rsSpec := newReplicaSet(2, labelMap)
	manager.rsStore.Store.Add(rsSpec)
	pods := newPodList(nil, 2, api.PodPending, labelMap, rsSpec)
	manager.podStore.Store.Add(&pods.Items[0])
	postExpectationsPod := pods.Items[1]

	manager.expectations = FakeRSExpectations{
		controller.NewControllerExpectations(), true, func() {
			// If we check active pods before checking expectataions, the
			// ReplicaSet will create a new replica because it doesn't see
			// this pod, but has fulfilled its expectations.
			manager.podStore.Store.Add(&postExpectationsPod)
		},
	}
	manager.syncReplicaSet(getKey(rsSpec, t))
	validateSyncReplicaSet(t, &fakePodControl, 0, 0)
}

func TestDeleteControllerAndExpectations(t *testing.T) {
	client := clientset.NewForConfigOrDie(&client.Config{Host: "", ContentConfig: client.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}})
	manager := NewReplicaSetController(client, controller.NoResyncPeriodFunc, 10, 0)
	manager.podStoreSynced = alwaysReady

	rs := newReplicaSet(1, map[string]string{"foo": "bar"})
	manager.rsStore.Store.Add(rs)

	fakePodControl := controller.FakePodControl{}
	manager.podControl = &fakePodControl

	// This should set expectations for the ReplicaSet
	manager.syncReplicaSet(getKey(rs, t))
	validateSyncReplicaSet(t, &fakePodControl, 1, 0)
	fakePodControl.Clear()

	// Get the ReplicaSet key
	rsKey, err := controller.KeyFunc(rs)
	if err != nil {
		t.Errorf("Couldn't get key for object %+v: %v", rs, err)
	}

	// This is to simulate a concurrent addPod, that has a handle on the expectations
	// as the controller deletes it.
	podExp, exists, err := manager.expectations.GetExpectations(rsKey)
	if !exists || err != nil {
		t.Errorf("No expectations found for ReplicaSet")
	}
	manager.rsStore.Delete(rs)
	manager.syncReplicaSet(getKey(rs, t))

	if _, exists, err = manager.expectations.GetExpectations(rsKey); exists {
		t.Errorf("Found expectaions, expected none since the ReplicaSet has been deleted.")
	}

	// This should have no effect, since we've deleted the ReplicaSet.
	podExp.Seen(1, 0)
	manager.podStore.Store.Replace(make([]interface{}, 0), "0")
	manager.syncReplicaSet(getKey(rs, t))
	validateSyncReplicaSet(t, &fakePodControl, 0, 0)
}

func TestRSManagerNotReady(t *testing.T) {
	client := clientset.NewForConfigOrDie(&client.Config{Host: "", ContentConfig: client.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}})
	fakePodControl := controller.FakePodControl{}
	manager := NewReplicaSetController(client, controller.NoResyncPeriodFunc, 2, 0)
	manager.podControl = &fakePodControl
	manager.podStoreSynced = func() bool { return false }

	// Simulates the ReplicaSet reflector running before the pod reflector. We don't
	// want to end up creating replicas in this case until the pod reflector
	// has synced, so the ReplicaSet controller should just requeue the ReplicaSet.
	rsSpec := newReplicaSet(1, map[string]string{"foo": "bar"})
	manager.rsStore.Store.Add(rsSpec)

	rsKey := getKey(rsSpec, t)
	manager.syncReplicaSet(rsKey)
	validateSyncReplicaSet(t, &fakePodControl, 0, 0)
	queueRS, _ := manager.queue.Get()
	if queueRS != rsKey {
		t.Fatalf("Expected to find key %v in queue, found %v", rsKey, queueRS)
	}

	manager.podStoreSynced = alwaysReady
	manager.syncReplicaSet(rsKey)
	validateSyncReplicaSet(t, &fakePodControl, 1, 0)
}

// shuffle returns a new shuffled list of container controllers.
func shuffle(controllers []*extensions.ReplicaSet) []*extensions.ReplicaSet {
	numControllers := len(controllers)
	randIndexes := rand.Perm(numControllers)
	shuffled := make([]*extensions.ReplicaSet, numControllers)
	for i := 0; i < numControllers; i++ {
		shuffled[i] = controllers[randIndexes[i]]
	}
	return shuffled
}

func TestOverlappingRSs(t *testing.T) {
	client := clientset.NewForConfigOrDie(&client.Config{Host: "", ContentConfig: client.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}})
	labelMap := map[string]string{"foo": "bar"}

	for i := 0; i < 5; i++ {
		manager := NewReplicaSetController(client, controller.NoResyncPeriodFunc, 10, 0)
		manager.podStoreSynced = alwaysReady

		// Create 10 ReplicaSets, shuffled them randomly and insert them into the ReplicaSet controller's store
		var controllers []*extensions.ReplicaSet
		for j := 1; j < 10; j++ {
			rsSpec := newReplicaSet(1, labelMap)
			rsSpec.CreationTimestamp = unversioned.Date(2014, time.December, j, 0, 0, 0, 0, time.Local)
			rsSpec.Name = string(util.NewUUID())
			controllers = append(controllers, rsSpec)
		}
		shuffledControllers := shuffle(controllers)
		for j := range shuffledControllers {
			manager.rsStore.Store.Add(shuffledControllers[j])
		}
		// Add a pod and make sure only the oldest ReplicaSet is synced
		pods := newPodList(nil, 1, api.PodPending, labelMap, controllers[0])
		rsKey := getKey(controllers[0], t)

		manager.addPod(&pods.Items[0])
		queueRS, _ := manager.queue.Get()
		if queueRS != rsKey {
			t.Fatalf("Expected to find key %v in queue, found %v", rsKey, queueRS)
		}
	}
}
