/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package controller

import (
	"fmt"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sort"
	"sync"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/cache"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/record"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/testclient"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/securitycontext"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/wait"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

type FakePodControl struct {
	controllerSpec []api.ReplicationController
	deletePodName  []string
	lock           sync.Mutex
	err            error
}

// Give each test that starts a background controller upto 1/2 a second.
// Since we need to start up a goroutine to test watch, this routine needs
// to get cpu before the test can complete. If the test is starved of cpu,
// the watch test will take upto 1/2 a second before timing out.
const controllerTimeout = 500 * time.Millisecond

var alwaysReady = func() bool { return true }

func init() {
	api.ForTesting_ReferencesAllowBlankSelfLinks = true
}

func (f *FakePodControl) createReplica(namespace string, spec *api.ReplicationController) error {
	f.lock.Lock()
	defer f.lock.Unlock()
	if f.err != nil {
		return f.err
	}
	f.controllerSpec = append(f.controllerSpec, *spec)
	return nil
}

func (f *FakePodControl) deletePod(namespace string, podName string) error {
	f.lock.Lock()
	defer f.lock.Unlock()
	if f.err != nil {
		return f.err
	}
	f.deletePodName = append(f.deletePodName, podName)
	return nil
}

func (f *FakePodControl) clear() {
	f.lock.Lock()
	defer f.lock.Unlock()
	f.deletePodName = []string{}
	f.controllerSpec = []api.ReplicationController{}
}

func getKey(rc *api.ReplicationController, t *testing.T) string {
	if key, err := rcKeyFunc(rc); err != nil {
		t.Errorf("Unexpected error getting key for rc %v: %v", rc.Name, err)
		return ""
	} else {
		return key
	}
}

func newReplicationController(replicas int) *api.ReplicationController {
	rc := &api.ReplicationController{
		TypeMeta: api.TypeMeta{APIVersion: testapi.Version()},
		ObjectMeta: api.ObjectMeta{
			UID:             util.NewUUID(),
			Name:            "foobar",
			Namespace:       api.NamespaceDefault,
			ResourceVersion: "18",
		},
		Spec: api.ReplicationControllerSpec{
			Replicas: replicas,
			Selector: map[string]string{"foo": "bar"},
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
	return rc
}

// create count pods with the given phase for the given rc (same selectors and namespace), and add them to the store.
func newPodList(store cache.Store, count int, status api.PodPhase, rc *api.ReplicationController) *api.PodList {
	pods := []api.Pod{}
	for i := 0; i < count; i++ {
		newPod := api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name:      fmt.Sprintf("pod%d", i),
				Labels:    rc.Spec.Selector,
				Namespace: rc.Namespace,
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

func validateSyncReplication(t *testing.T, fakePodControl *FakePodControl, expectedCreates, expectedDeletes int) {
	if len(fakePodControl.controllerSpec) != expectedCreates {
		t.Errorf("Unexpected number of creates.  Expected %d, saw %d\n", expectedCreates, len(fakePodControl.controllerSpec))
	}
	if len(fakePodControl.deletePodName) != expectedDeletes {
		t.Errorf("Unexpected number of deletes.  Expected %d, saw %d\n", expectedDeletes, len(fakePodControl.deletePodName))
	}
}

func replicationControllerResourceName() string {
	return "replicationcontrollers"
}

type serverResponse struct {
	statusCode int
	obj        interface{}
}

func makeTestServer(t *testing.T, namespace, name string, podResponse, controllerResponse, updateResponse serverResponse) (*httptest.Server, *util.FakeHandler) {
	fakePodHandler := util.FakeHandler{
		StatusCode:   podResponse.statusCode,
		ResponseBody: runtime.EncodeOrDie(testapi.Codec(), podResponse.obj.(runtime.Object)),
	}
	fakeControllerHandler := util.FakeHandler{
		StatusCode:   controllerResponse.statusCode,
		ResponseBody: runtime.EncodeOrDie(testapi.Codec(), controllerResponse.obj.(runtime.Object)),
	}
	fakeUpdateHandler := util.FakeHandler{
		StatusCode:   updateResponse.statusCode,
		ResponseBody: runtime.EncodeOrDie(testapi.Codec(), updateResponse.obj.(runtime.Object)),
	}
	mux := http.NewServeMux()
	mux.Handle(testapi.ResourcePath("pods", namespace, ""), &fakePodHandler)
	mux.Handle(testapi.ResourcePath(replicationControllerResourceName(), "", ""), &fakeControllerHandler)
	if namespace != "" {
		mux.Handle(testapi.ResourcePath(replicationControllerResourceName(), namespace, ""), &fakeControllerHandler)
	}
	if name != "" {
		mux.Handle(testapi.ResourcePath(replicationControllerResourceName(), namespace, name), &fakeUpdateHandler)
	}
	mux.HandleFunc("/", func(res http.ResponseWriter, req *http.Request) {
		t.Errorf("unexpected request: %v", req.RequestURI)
		res.WriteHeader(http.StatusNotFound)
	})
	return httptest.NewServer(mux), &fakeUpdateHandler
}

func startManagerAndWait(manager *ReplicationManager, pods int, t *testing.T) chan struct{} {
	stopCh := make(chan struct{})
	go manager.Run(1, stopCh)
	err := wait.Poll(10*time.Millisecond, 100*time.Millisecond, func() (bool, error) {
		podList, err := manager.podStore.List(labels.Everything())
		if err != nil {
			return false, err
		}
		return len(podList) == pods, nil
	})
	if err != nil {
		t.Errorf("Failed to observe %d pods in 100ms", pods)
	}
	return stopCh
}

func TestSyncReplicationControllerDoesNothing(t *testing.T) {
	client := client.NewOrDie(&client.Config{Host: "", Version: testapi.Version()})
	fakePodControl := FakePodControl{}
	manager := NewReplicationManager(client, BurstReplicas)
	manager.podStoreSynced = alwaysReady

	// 2 running pods, a controller with 2 replicas, sync is a no-op
	controllerSpec := newReplicationController(2)
	manager.controllerStore.Store.Add(controllerSpec)
	newPodList(manager.podStore.Store, 2, api.PodRunning, controllerSpec)

	manager.podControl = &fakePodControl
	manager.syncReplicationController(getKey(controllerSpec, t))
	validateSyncReplication(t, &fakePodControl, 0, 0)
}

func TestSyncReplicationControllerDeletes(t *testing.T) {
	client := client.NewOrDie(&client.Config{Host: "", Version: testapi.Version()})
	fakePodControl := FakePodControl{}
	manager := NewReplicationManager(client, BurstReplicas)
	manager.podStoreSynced = alwaysReady
	manager.podControl = &fakePodControl

	// 2 running pods and a controller with 1 replica, one pod delete expected
	controllerSpec := newReplicationController(1)
	manager.controllerStore.Store.Add(controllerSpec)
	newPodList(manager.podStore.Store, 2, api.PodRunning, controllerSpec)

	manager.syncReplicationController(getKey(controllerSpec, t))
	validateSyncReplication(t, &fakePodControl, 0, 1)
}

func TestDeleteFinalStateUnknown(t *testing.T) {
	client := client.NewOrDie(&client.Config{Host: "", Version: testapi.Version()})
	fakePodControl := FakePodControl{}
	manager := NewReplicationManager(client, BurstReplicas)
	manager.podStoreSynced = alwaysReady
	manager.podControl = &fakePodControl

	received := make(chan string)
	manager.syncHandler = func(key string) error {
		received <- key
		return nil
	}

	// The DeletedFinalStateUnknown object should cause the rc manager to insert
	// the controller matching the selectors of the deleted pod into the work queue.
	controllerSpec := newReplicationController(1)
	manager.controllerStore.Store.Add(controllerSpec)
	pods := newPodList(nil, 1, api.PodRunning, controllerSpec)
	manager.deletePod(cache.DeletedFinalStateUnknown{Key: "foo", Obj: &pods.Items[0]})

	go manager.worker()

	expected := getKey(controllerSpec, t)
	select {
	case key := <-received:
		if key != expected {
			t.Errorf("Unexpected sync all for rc %v, expected %v", key, expected)
		}
	case <-time.After(100 * time.Millisecond):
		t.Errorf("Processing DeleteFinalStateUnknown took longer than expected")
	}
}

func TestSyncReplicationControllerCreates(t *testing.T) {
	client := client.NewOrDie(&client.Config{Host: "", Version: testapi.Version()})
	manager := NewReplicationManager(client, BurstReplicas)
	manager.podStoreSynced = alwaysReady

	// A controller with 2 replicas and no pods in the store, 2 creates expected
	controller := newReplicationController(2)
	manager.controllerStore.Store.Add(controller)

	fakePodControl := FakePodControl{}
	manager.podControl = &fakePodControl
	manager.syncReplicationController(getKey(controller, t))
	validateSyncReplication(t, &fakePodControl, 2, 0)
}

func TestCreateReplica(t *testing.T) {
	ns := api.NamespaceDefault
	body := runtime.EncodeOrDie(testapi.Codec(), &api.Pod{ObjectMeta: api.ObjectMeta{Name: "empty_pod"}})
	fakeHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(body),
	}
	testServer := httptest.NewServer(&fakeHandler)
	defer testServer.Close()
	client := client.NewOrDie(&client.Config{Host: testServer.URL, Version: testapi.Version()})

	podControl := RealPodControl{
		kubeClient: client,
		recorder:   &record.FakeRecorder{},
	}

	controllerSpec := newReplicationController(1)

	// Make sure createReplica sends a POST to the apiserver with a pod from the controllers pod template
	podControl.createReplica(ns, controllerSpec)

	expectedPod := api.Pod{
		ObjectMeta: api.ObjectMeta{
			Labels:       controllerSpec.Spec.Template.Labels,
			GenerateName: fmt.Sprintf("%s-", controllerSpec.Name),
		},
		Spec: controllerSpec.Spec.Template.Spec,
	}
	fakeHandler.ValidateRequest(t, testapi.ResourcePath("pods", api.NamespaceDefault, ""), "POST", nil)
	actualPod, err := client.Codec.Decode([]byte(fakeHandler.RequestBody))
	if err != nil {
		t.Errorf("Unexpected error: %#v", err)
	}
	if !api.Semantic.DeepDerivative(&expectedPod, actualPod) {
		t.Logf("Body: %s", fakeHandler.RequestBody)
		t.Errorf("Unexpected mismatch.  Expected\n %#v,\n Got:\n %#v", &expectedPod, actualPod)
	}
}

func TestStatusUpdatesWithoutReplicasChange(t *testing.T) {
	// Setup a fake server to listen for requests, and run the rc manager in steady state
	fakeHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: "",
	}
	testServer := httptest.NewServer(&fakeHandler)
	defer testServer.Close()
	client := client.NewOrDie(&client.Config{Host: testServer.URL, Version: testapi.Version()})
	manager := NewReplicationManager(client, BurstReplicas)
	manager.podStoreSynced = alwaysReady

	// Steady state for the replication controller, no Status.Replicas updates expected
	activePods := 5
	rc := newReplicationController(activePods)
	manager.controllerStore.Store.Add(rc)
	rc.Status = api.ReplicationControllerStatus{Replicas: activePods}
	newPodList(manager.podStore.Store, activePods, api.PodRunning, rc)

	fakePodControl := FakePodControl{}
	manager.podControl = &fakePodControl
	manager.syncReplicationController(getKey(rc, t))

	validateSyncReplication(t, &fakePodControl, 0, 0)
	if fakeHandler.RequestReceived != nil {
		t.Errorf("Unexpected update when pods and rcs are in a steady state")
	}

	// This response body is just so we don't err out decoding the http response, all
	// we care about is the request body sent below.
	response := runtime.EncodeOrDie(testapi.Codec(), &api.ReplicationController{})
	fakeHandler.ResponseBody = response

	rc.Generation = rc.Generation + 1
	manager.syncReplicationController(getKey(rc, t))

	rc.Status.ObservedGeneration = rc.Generation
	updatedRc := runtime.EncodeOrDie(testapi.Codec(), rc)
	fakeHandler.ValidateRequest(t, testapi.ResourcePath(replicationControllerResourceName(), rc.Namespace, rc.Name), "PUT", &updatedRc)
}

func TestControllerUpdateReplicas(t *testing.T) {
	// This is a happy server just to record the PUT request we expect for status.Replicas
	fakeHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: "",
	}
	testServer := httptest.NewServer(&fakeHandler)
	defer testServer.Close()

	client := client.NewOrDie(&client.Config{Host: testServer.URL, Version: testapi.Version()})
	manager := NewReplicationManager(client, BurstReplicas)
	manager.podStoreSynced = alwaysReady

	// Insufficient number of pods in the system, and Status.Replicas is wrong;
	// Status.Replica should update to match number of pods in system, 1 new pod should be created.
	rc := newReplicationController(5)
	manager.controllerStore.Store.Add(rc)
	rc.Status = api.ReplicationControllerStatus{Replicas: 2, ObservedGeneration: 0}
	rc.Generation = 1
	newPodList(manager.podStore.Store, 4, api.PodRunning, rc)

	// This response body is just so we don't err out decoding the http response
	response := runtime.EncodeOrDie(testapi.Codec(), &api.ReplicationController{})
	fakeHandler.ResponseBody = response

	fakePodControl := FakePodControl{}
	manager.podControl = &fakePodControl

	manager.syncReplicationController(getKey(rc, t))

	// 1. Status.Replicas should go up from 2->4 even though we created 5-4=1 pod.
	// 2. Every update to the status should include the Generation of the spec.
	rc.Status = api.ReplicationControllerStatus{Replicas: 4, ObservedGeneration: 1}

	decRc := runtime.EncodeOrDie(testapi.Codec(), rc)
	fakeHandler.ValidateRequest(t, testapi.ResourcePath(replicationControllerResourceName(), rc.Namespace, rc.Name), "PUT", &decRc)
	validateSyncReplication(t, &fakePodControl, 1, 0)
}

func TestActivePodFiltering(t *testing.T) {
	// This rc is not needed by the test, only the newPodList to give the pods labels/a namespace.
	rc := newReplicationController(0)
	podList := newPodList(nil, 5, api.PodRunning, rc)
	podList.Items[0].Status.Phase = api.PodSucceeded
	podList.Items[1].Status.Phase = api.PodFailed
	expectedNames := util.NewStringSet()
	for _, pod := range podList.Items[2:] {
		expectedNames.Insert(pod.Name)
	}

	got := filterActivePods(podList.Items)
	gotNames := util.NewStringSet()
	for _, pod := range got {
		gotNames.Insert(pod.Name)
	}
	if expectedNames.Difference(gotNames).Len() != 0 || gotNames.Difference(expectedNames).Len() != 0 {
		t.Errorf("expected %v, got %v", expectedNames.List(), gotNames.List())
	}
}

func TestSortingActivePods(t *testing.T) {
	numPods := 5
	// This rc is not needed by the test, only the newPodList to give the pods labels/a namespace.
	rc := newReplicationController(0)
	podList := newPodList(nil, numPods, api.PodRunning, rc)

	pods := make([]*api.Pod, len(podList.Items))
	for i := range podList.Items {
		pods[i] = &podList.Items[i]
	}
	// pods[0] is not scheduled yet.
	pods[0].Spec.NodeName = ""
	pods[0].Status.Phase = api.PodPending
	// pods[1] is scheduled but pending.
	pods[1].Spec.NodeName = "bar"
	pods[1].Status.Phase = api.PodPending
	// pods[2] is unknown.
	pods[2].Spec.NodeName = "foo"
	pods[2].Status.Phase = api.PodUnknown
	// pods[3] is running but not ready.
	pods[3].Spec.NodeName = "foo"
	pods[3].Status.Phase = api.PodRunning
	// pods[4] is running and ready.
	pods[4].Spec.NodeName = "foo"
	pods[4].Status.Phase = api.PodRunning
	pods[4].Status.Conditions = []api.PodCondition{{Type: api.PodReady, Status: api.ConditionTrue}}

	getOrder := func(pods []*api.Pod) []string {
		names := make([]string, len(pods))
		for i := range pods {
			names[i] = pods[i].Name
		}
		return names
	}

	expected := getOrder(pods)

	for i := 0; i < 20; i++ {
		idx := rand.Perm(numPods)
		randomizedPods := make([]*api.Pod, numPods)
		for j := 0; j < numPods; j++ {
			randomizedPods[j] = pods[idx[j]]
		}
		sort.Sort(activePods(randomizedPods))
		actual := getOrder(randomizedPods)

		if !reflect.DeepEqual(actual, expected) {
			t.Errorf("expected %v, got %v", expected, actual)
		}
	}
}

// NewFakeRCExpectationsLookup creates a fake store for PodExpectations.
func NewFakeRCExpectationsLookup(ttl time.Duration) (*RCExpectations, *util.FakeClock) {
	fakeTime := time.Date(2009, time.November, 10, 23, 0, 0, 0, time.UTC)
	fakeClock := &util.FakeClock{fakeTime}
	ttlPolicy := &cache.TTLPolicy{ttl, fakeClock}
	ttlStore := cache.NewFakeExpirationStore(
		expKeyFunc, nil, ttlPolicy, fakeClock)
	return &RCExpectations{ttlStore}, fakeClock
}

func TestRCExpectations(t *testing.T) {
	ttl := 30 * time.Second
	e, fakeClock := NewFakeRCExpectationsLookup(ttl)
	// In practice we can't really have add and delete expectations since we only either create or
	// delete replicas in one rc pass, and the rc goes to sleep soon after until the expectations are
	// either fulfilled or timeout.
	adds, dels := 10, 30
	rc := newReplicationController(1)

	// RC fires off adds and deletes at apiserver, then sets expectations
	e.setExpectations(rc, adds, dels)
	var wg sync.WaitGroup
	for i := 0; i < adds+1; i++ {
		wg.Add(1)
		go func() {
			// In prod this can happen either because of a failed create by the rc
			// or after having observed a create via informer
			e.CreationObserved(rc)
			wg.Done()
		}()
	}
	wg.Wait()

	// There are still delete expectations
	if e.SatisfiedExpectations(rc) {
		t.Errorf("Rc will sync before expectations are met")
	}
	for i := 0; i < dels+1; i++ {
		wg.Add(1)
		go func() {
			e.DeletionObserved(rc)
			wg.Done()
		}()
	}
	wg.Wait()

	// Expectations have been surpassed
	if podExp, exists, err := e.GetExpectations(rc); err == nil && exists {
		add, del := podExp.getExpectations()
		if add != -1 || del != -1 {
			t.Errorf("Unexpected pod expectations %#v", podExp)
		}
	} else {
		t.Errorf("Could not get expectations for rc, exists %v and err %v", exists, err)
	}
	if !e.SatisfiedExpectations(rc) {
		t.Errorf("Expectations are met but the rc will not sync")
	}

	// Next round of rc sync, old expectations are cleared
	e.setExpectations(rc, 1, 2)
	if podExp, exists, err := e.GetExpectations(rc); err == nil && exists {
		add, del := podExp.getExpectations()
		if add != 1 || del != 2 {
			t.Errorf("Unexpected pod expectations %#v", podExp)
		}
	} else {
		t.Errorf("Could not get expectations for rc, exists %v and err %v", exists, err)
	}

	// Expectations have expired because of ttl
	fakeClock.Time = fakeClock.Time.Add(ttl + 1)
	if !e.SatisfiedExpectations(rc) {
		t.Errorf("Expectations should have expired but didn't")
	}
}

func TestSyncReplicationControllerDormancy(t *testing.T) {
	// Setup a test server so we can lie about the current state of pods
	fakeHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: "",
	}
	testServer := httptest.NewServer(&fakeHandler)
	defer testServer.Close()
	client := client.NewOrDie(&client.Config{Host: testServer.URL, Version: testapi.Version()})

	fakePodControl := FakePodControl{}
	manager := NewReplicationManager(client, BurstReplicas)
	manager.podStoreSynced = alwaysReady
	manager.podControl = &fakePodControl

	controllerSpec := newReplicationController(2)
	manager.controllerStore.Store.Add(controllerSpec)
	newPodList(manager.podStore.Store, 1, api.PodRunning, controllerSpec)

	// Creates a replica and sets expectations
	controllerSpec.Status.Replicas = 1
	manager.syncReplicationController(getKey(controllerSpec, t))
	validateSyncReplication(t, &fakePodControl, 1, 0)

	// Expectations prevents replicas but not an update on status
	controllerSpec.Status.Replicas = 0
	fakePodControl.clear()
	manager.syncReplicationController(getKey(controllerSpec, t))
	validateSyncReplication(t, &fakePodControl, 0, 0)

	// Lowering expectations should lead to a sync that creates a replica, however the
	// fakePodControl error will prevent this, leaving expectations at 0, 0
	manager.expectations.CreationObserved(controllerSpec)
	controllerSpec.Status.Replicas = 1
	fakePodControl.clear()
	fakePodControl.err = fmt.Errorf("Fake Error")

	manager.syncReplicationController(getKey(controllerSpec, t))
	validateSyncReplication(t, &fakePodControl, 0, 0)

	// This replica should not need a Lowering of expectations, since the previous create failed
	fakePodControl.err = nil
	manager.syncReplicationController(getKey(controllerSpec, t))
	validateSyncReplication(t, &fakePodControl, 1, 0)

	// 1 PUT for the rc status during dormancy window.
	// Note that the pod creates go through pod control so they're not recorded.
	fakeHandler.ValidateRequestCount(t, 1)
}

func TestPodControllerLookup(t *testing.T) {
	manager := NewReplicationManager(client.NewOrDie(&client.Config{Host: "", Version: testapi.Version()}), BurstReplicas)
	manager.podStoreSynced = alwaysReady
	testCases := []struct {
		inRCs     []*api.ReplicationController
		pod       *api.Pod
		outRCName string
	}{
		// pods without labels don't match any rcs
		{
			inRCs: []*api.ReplicationController{
				{ObjectMeta: api.ObjectMeta{Name: "basic"}}},
			pod:       &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo1", Namespace: api.NamespaceAll}},
			outRCName: "",
		},
		// Matching labels, not namespace
		{
			inRCs: []*api.ReplicationController{
				{
					ObjectMeta: api.ObjectMeta{Name: "foo"},
					Spec: api.ReplicationControllerSpec{
						Selector: map[string]string{"foo": "bar"},
					},
				},
			},
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "foo2", Namespace: "ns", Labels: map[string]string{"foo": "bar"}}},
			outRCName: "",
		},
		// Matching ns and labels returns the key to the rc, not the rc name
		{
			inRCs: []*api.ReplicationController{
				{
					ObjectMeta: api.ObjectMeta{Name: "bar", Namespace: "ns"},
					Spec: api.ReplicationControllerSpec{
						Selector: map[string]string{"foo": "bar"},
					},
				},
			},
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "foo3", Namespace: "ns", Labels: map[string]string{"foo": "bar"}}},
			outRCName: "bar",
		},
	}
	for _, c := range testCases {
		for _, r := range c.inRCs {
			manager.controllerStore.Add(r)
		}
		if rc := manager.getPodControllers(c.pod); rc != nil {
			if c.outRCName != rc.Name {
				t.Errorf("Got controller %+v expected %+v", rc.Name, c.outRCName)
			}
		} else if c.outRCName != "" {
			t.Errorf("Expected a controller %v pod %v, found none", c.outRCName, c.pod.Name)
		}
	}
}

type FakeWatcher struct {
	w *watch.FakeWatcher
	*testclient.Fake
}

func TestWatchControllers(t *testing.T) {
	fakeWatch := watch.NewFake()
	client := &testclient.Fake{Watch: fakeWatch}
	manager := NewReplicationManager(client, BurstReplicas)
	manager.podStoreSynced = alwaysReady

	var testControllerSpec api.ReplicationController
	received := make(chan string)

	// The update sent through the fakeWatcher should make its way into the workqueue,
	// and eventually into the syncHandler. The handler validates the received controller
	// and closes the received channel to indicate that the test can finish.
	manager.syncHandler = func(key string) error {

		obj, exists, err := manager.controllerStore.Store.GetByKey(key)
		if !exists || err != nil {
			t.Errorf("Expected to find controller under key %v", key)
		}
		controllerSpec := *obj.(*api.ReplicationController)
		if !api.Semantic.DeepDerivative(controllerSpec, testControllerSpec) {
			t.Errorf("Expected %#v, but got %#v", testControllerSpec, controllerSpec)
		}
		close(received)
		return nil
	}
	// Start only the rc watcher and the workqueue, send a watch event,
	// and make sure it hits the sync method.
	stopCh := make(chan struct{})
	defer close(stopCh)
	go manager.rcController.Run(stopCh)
	go util.Until(manager.worker, 10*time.Millisecond, stopCh)

	testControllerSpec.Name = "foo"
	fakeWatch.Add(&testControllerSpec)

	select {
	case <-received:
	case <-time.After(controllerTimeout):
		t.Errorf("Expected 1 call but got 0")
	}
}

func TestWatchPods(t *testing.T) {
	fakeWatch := watch.NewFake()
	client := &testclient.Fake{Watch: fakeWatch}
	manager := NewReplicationManager(client, BurstReplicas)
	manager.podStoreSynced = alwaysReady

	// Put one rc and one pod into the controller's stores
	testControllerSpec := newReplicationController(1)
	manager.controllerStore.Store.Add(testControllerSpec)
	received := make(chan string)
	// The pod update sent through the fakeWatcher should figure out the managing rc and
	// send it into the syncHandler.
	manager.syncHandler = func(key string) error {

		obj, exists, err := manager.controllerStore.Store.GetByKey(key)
		if !exists || err != nil {
			t.Errorf("Expected to find controller under key %v", key)
		}
		controllerSpec := obj.(*api.ReplicationController)
		if !api.Semantic.DeepDerivative(controllerSpec, testControllerSpec) {
			t.Errorf("\nExpected %#v,\nbut got %#v", testControllerSpec, controllerSpec)
		}
		close(received)
		return nil
	}
	// Start only the pod watcher and the workqueue, send a watch event,
	// and make sure it hits the sync method for the right rc.
	stopCh := make(chan struct{})
	defer close(stopCh)
	go manager.podController.Run(stopCh)
	go util.Until(manager.worker, 10*time.Millisecond, stopCh)

	pods := newPodList(nil, 1, api.PodRunning, testControllerSpec)
	testPod := pods.Items[0]
	testPod.Status.Phase = api.PodFailed
	fakeWatch.Add(&testPod)

	select {
	case <-received:
	case <-time.After(controllerTimeout):
		t.Errorf("Expected 1 call but got 0")
	}
}

func TestUpdatePods(t *testing.T) {
	fakeWatch := watch.NewFake()
	client := &testclient.Fake{Watch: fakeWatch}
	manager := NewReplicationManager(client, BurstReplicas)
	manager.podStoreSynced = alwaysReady

	received := make(chan string)

	manager.syncHandler = func(key string) error {
		obj, exists, err := manager.controllerStore.Store.GetByKey(key)
		if !exists || err != nil {
			t.Errorf("Expected to find controller under key %v", key)
		}
		received <- obj.(*api.ReplicationController).Name
		return nil
	}

	stopCh := make(chan struct{})
	defer close(stopCh)
	go util.Until(manager.worker, 10*time.Millisecond, stopCh)

	// Put 2 rcs and one pod into the controller's stores
	testControllerSpec1 := newReplicationController(1)
	manager.controllerStore.Store.Add(testControllerSpec1)
	testControllerSpec2 := *testControllerSpec1
	testControllerSpec2.Spec.Selector = map[string]string{"bar": "foo"}
	testControllerSpec2.Name = "barfoo"
	manager.controllerStore.Store.Add(&testControllerSpec2)

	// Put one pod in the podStore
	pod1 := newPodList(manager.podStore.Store, 1, api.PodRunning, testControllerSpec1).Items[0]
	pod2 := pod1
	pod2.Labels = testControllerSpec2.Spec.Selector

	// Send an update of the same pod with modified labels, and confirm we get a sync request for
	// both controllers
	manager.updatePod(&pod1, &pod2)

	expected := util.NewStringSet(testControllerSpec1.Name, testControllerSpec2.Name)
	for _, name := range expected.List() {
		t.Logf("Expecting update for %+v", name)
		select {
		case got := <-received:
			if !expected.Has(got) {
				t.Errorf("Expected keys %#v got %v", expected, got)
			}
		case <-time.After(controllerTimeout):
			t.Errorf("Expected update notifications for controllers within 100ms each")
		}
	}
}

func TestControllerUpdateRequeue(t *testing.T) {
	// This server should force a requeue of the controller becuase it fails to update status.Replicas.
	fakeHandler := util.FakeHandler{
		StatusCode:   500,
		ResponseBody: "",
	}
	testServer := httptest.NewServer(&fakeHandler)
	defer testServer.Close()

	client := client.NewOrDie(&client.Config{Host: testServer.URL, Version: testapi.Version()})
	manager := NewReplicationManager(client, BurstReplicas)
	manager.podStoreSynced = alwaysReady

	rc := newReplicationController(1)
	manager.controllerStore.Store.Add(rc)
	rc.Status = api.ReplicationControllerStatus{Replicas: 2}
	newPodList(manager.podStore.Store, 1, api.PodRunning, rc)

	fakePodControl := FakePodControl{}
	manager.podControl = &fakePodControl

	manager.syncReplicationController(getKey(rc, t))

	ch := make(chan interface{})
	go func() {
		item, _ := manager.queue.Get()
		ch <- item
	}()
	select {
	case key := <-ch:
		expectedKey := getKey(rc, t)
		if key != expectedKey {
			t.Errorf("Expected requeue of controller with key %s got %s", expectedKey, key)
		}
	case <-time.After(controllerTimeout):
		manager.queue.ShutDown()
		t.Errorf("Expected to find an rc in the queue, found none.")
	}
	// 1 Update and 1 GET, both of which fail
	fakeHandler.ValidateRequestCount(t, 2)
}

func TestControllerUpdateStatusWithFailure(t *testing.T) {
	rc := newReplicationController(1)
	fakeClient := &testclient.Fake{
		ReactFn: func(f testclient.FakeAction) (runtime.Object, error) {
			if f.Action == testclient.GetControllerAction {
				return rc, nil
			}
			return &api.ReplicationController{}, fmt.Errorf("Fake error")
		},
	}
	fakeRCClient := &testclient.FakeReplicationControllers{fakeClient, "default"}
	numReplicas := 10
	updateReplicaCount(fakeRCClient, *rc, numReplicas)
	updates, gets := 0, 0
	for _, a := range fakeClient.Actions {
		switch a.Action {
		case testclient.GetControllerAction:
			gets++
			// Make sure the get is for the right rc even though the update failed.
			if s, ok := a.Value.(string); !ok || s != rc.Name {
				t.Errorf("Expected get for rc %v, got %+v instead", rc.Name, s)
			}
		case testclient.UpdateControllerAction:
			updates++
			// Confirm that the update has the right status.Replicas even though the Get
			// returned an rc with replicas=1.
			if c, ok := a.Value.(*api.ReplicationController); !ok {
				t.Errorf("Expected an rc as the argument to update, got %T", c)
			} else if c.Status.Replicas != numReplicas {
				t.Errorf("Expected update for rc to contain replicas %v, got %v instead",
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
	client := client.NewOrDie(&client.Config{Host: "", Version: testapi.Version()})
	fakePodControl := FakePodControl{}
	manager := NewReplicationManager(client, burstReplicas)
	manager.podStoreSynced = alwaysReady
	manager.podControl = &fakePodControl

	controllerSpec := newReplicationController(numReplicas)
	manager.controllerStore.Store.Add(controllerSpec)

	expectedPods := 0
	pods := newPodList(nil, numReplicas, api.PodPending, controllerSpec)

	// Size up the controller, then size it down, and confirm the expected create/delete pattern
	for _, replicas := range []int{numReplicas, 0} {

		controllerSpec.Spec.Replicas = replicas
		manager.controllerStore.Store.Add(controllerSpec)

		for i := 0; i < numReplicas; i += burstReplicas {
			manager.syncReplicationController(getKey(controllerSpec, t))

			// The store accrues active pods. It's also used by the rc to determine how many
			// replicas to create.
			activePods := len(manager.podStore.Store.List())
			if replicas != 0 {
				// This is the number of pods currently "in flight". They were created by the rc manager above,
				// which then puts the rc to sleep till all of them have been observed.
				expectedPods = replicas - activePods
				if expectedPods > burstReplicas {
					expectedPods = burstReplicas
				}
				// This validates the rc manager sync actually created pods
				validateSyncReplication(t, &fakePodControl, expectedPods, 0)

				// This simulates the watch events for all but 1 of the expected pods.
				// None of these should wake the controller because it has expectations==BurstReplicas.
				for i := 0; i < expectedPods-1; i++ {
					manager.podStore.Store.Add(&pods.Items[i])
					manager.addPod(&pods.Items[i])
				}

				podExp, exists, err := manager.expectations.GetExpectations(controllerSpec)
				if !exists || err != nil {
					t.Fatalf("Did not find expectations for rc.")
				}
				if add, _ := podExp.getExpectations(); add != 1 {
					t.Fatalf("Expectations are wrong %v", podExp)
				}
			} else {
				expectedPods = (replicas - activePods) * -1
				if expectedPods > burstReplicas {
					expectedPods = burstReplicas
				}
				validateSyncReplication(t, &fakePodControl, 0, expectedPods)
				for i := 0; i < expectedPods-1; i++ {
					manager.podStore.Store.Delete(&pods.Items[i])
					manager.deletePod(&pods.Items[i])
				}
				podExp, exists, err := manager.expectations.GetExpectations(controllerSpec)
				if !exists || err != nil {
					t.Fatalf("Did not find expectations for rc.")
				}
				if _, del := podExp.getExpectations(); del != 1 {
					t.Fatalf("Expectations are wrong %v", podExp)
				}
			}

			// Check that the rc didn't take any action for all the above pods
			fakePodControl.clear()
			manager.syncReplicationController(getKey(controllerSpec, t))
			validateSyncReplication(t, &fakePodControl, 0, 0)

			// Create/Delete the last pod
			// The last add pod will decrease the expectation of the rc to 0,
			// which will cause it to create/delete the remaining replicas upto burstReplicas.
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
		if activePods != controllerSpec.Spec.Replicas {
			t.Fatalf("Unexpected number of active pods, expected %d, got %d", controllerSpec.Spec.Replicas, activePods)
		}
		// Replenish the pod list, since we cut it down sizing up
		pods = newPodList(nil, replicas, api.PodRunning, controllerSpec)
	}
}

func TestControllerBurstReplicas(t *testing.T) {
	doTestControllerBurstReplicas(t, 5, 30)
	doTestControllerBurstReplicas(t, 5, 12)
	doTestControllerBurstReplicas(t, 3, 2)
}

type FakeRCExpectations struct {
	*RCExpectations
	satisfied    bool
	expSatisfied func()
}

func (fe FakeRCExpectations) SatisfiedExpectations(rc *api.ReplicationController) bool {
	fe.expSatisfied()
	return fe.satisfied
}

// TestRCSyncExpectations tests that a pod cannot sneak in between counting active pods
// and checking expectations.
func TestRCSyncExpectations(t *testing.T) {
	client := client.NewOrDie(&client.Config{Host: "", Version: testapi.Version()})
	fakePodControl := FakePodControl{}
	manager := NewReplicationManager(client, 2)
	manager.podStoreSynced = alwaysReady
	manager.podControl = &fakePodControl

	controllerSpec := newReplicationController(2)
	manager.controllerStore.Store.Add(controllerSpec)
	pods := newPodList(nil, 2, api.PodPending, controllerSpec)
	manager.podStore.Store.Add(&pods.Items[0])
	postExpectationsPod := pods.Items[1]

	manager.expectations = FakeRCExpectations{
		NewRCExpectations(), true, func() {
			// If we check active pods before checking expectataions, the rc
			// will create a new replica because it doesn't see this pod, but
			// has fulfilled its expectations.
			manager.podStore.Store.Add(&postExpectationsPod)
		},
	}
	manager.syncReplicationController(getKey(controllerSpec, t))
	validateSyncReplication(t, &fakePodControl, 0, 0)
}

func TestDeleteControllerAndExpectations(t *testing.T) {
	client := client.NewOrDie(&client.Config{Host: "", Version: testapi.Version()})
	manager := NewReplicationManager(client, 10)
	manager.podStoreSynced = alwaysReady

	rc := newReplicationController(1)
	manager.controllerStore.Store.Add(rc)

	fakePodControl := FakePodControl{}
	manager.podControl = &fakePodControl

	// This should set expectations for the rc
	manager.syncReplicationController(getKey(rc, t))
	validateSyncReplication(t, &fakePodControl, 1, 0)
	fakePodControl.clear()

	// This is to simulate a concurrent addPod, that has a handle on the expectations
	// as the controller deletes it.
	podExp, exists, err := manager.expectations.GetExpectations(rc)
	if !exists || err != nil {
		t.Errorf("No expectations found for rc")
	}
	manager.controllerStore.Delete(rc)
	manager.syncReplicationController(getKey(rc, t))

	if _, exists, err = manager.expectations.GetExpectations(rc); exists {
		t.Errorf("Found expectaions, expected none since the rc has been deleted.")
	}

	// This should have no effect, since we've deleted the rc.
	podExp.Seen(1, 0)
	manager.podStore.Store.Replace(make([]interface{}, 0))
	manager.syncReplicationController(getKey(rc, t))
	validateSyncReplication(t, &fakePodControl, 0, 0)
}

func TestRCManagerNotReady(t *testing.T) {
	client := client.NewOrDie(&client.Config{Host: "", Version: testapi.Version()})
	fakePodControl := FakePodControl{}
	manager := NewReplicationManager(client, 2)
	manager.podControl = &fakePodControl
	manager.podStoreSynced = func() bool { return false }

	// Simulates the rc reflector running before the pod reflector. We don't
	// want to end up creating replicas in this case until the pod reflector
	// has synced, so the rc manager should just requeue the rc.
	controllerSpec := newReplicationController(1)
	manager.controllerStore.Store.Add(controllerSpec)

	rcKey := getKey(controllerSpec, t)
	manager.syncReplicationController(rcKey)
	validateSyncReplication(t, &fakePodControl, 0, 0)
	queueRC, _ := manager.queue.Get()
	if queueRC != rcKey {
		t.Fatalf("Expected to find key %v in queue, found %v", rcKey, queueRC)
	}

	manager.podStoreSynced = alwaysReady
	manager.syncReplicationController(rcKey)
	validateSyncReplication(t, &fakePodControl, 1, 0)
}

// shuffle returns a new shuffled list of container controllers.
func shuffle(controllers []*api.ReplicationController) []*api.ReplicationController {
	numControllers := len(controllers)
	randIndexes := rand.Perm(numControllers)
	shuffled := make([]*api.ReplicationController, numControllers)
	for i := 0; i < numControllers; i++ {
		shuffled[i] = controllers[randIndexes[i]]
	}
	return shuffled
}

func TestOverlappingRCs(t *testing.T) {
	client := client.NewOrDie(&client.Config{Host: "", Version: testapi.Version()})

	for i := 0; i < 5; i++ {
		manager := NewReplicationManager(client, 10)
		manager.podStoreSynced = alwaysReady

		// Create 10 rcs, shuffled them randomly and insert them into the rc manager's store
		var controllers []*api.ReplicationController
		for j := 1; j < 10; j++ {
			controllerSpec := newReplicationController(1)
			controllerSpec.CreationTimestamp = util.Date(2014, time.December, j, 0, 0, 0, 0, time.Local)
			controllerSpec.Name = string(util.NewUUID())
			controllers = append(controllers, controllerSpec)
		}
		shuffledControllers := shuffle(controllers)
		for j := range shuffledControllers {
			manager.controllerStore.Store.Add(shuffledControllers[j])
		}
		// Add a pod and make sure only the oldest rc is synced
		pods := newPodList(nil, 1, api.PodPending, controllers[0])
		rcKey := getKey(controllers[0], t)

		manager.addPod(&pods.Items[0])
		queueRC, _ := manager.queue.Get()
		if queueRC != rcKey {
			t.Fatalf("Expected to find key %v in queue, found %v", rcKey, queueRC)
		}
	}
}
