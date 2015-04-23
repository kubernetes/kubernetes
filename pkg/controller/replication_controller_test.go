/*
Copyright 2014 Google Inc. All rights reserved.

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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/testclient"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

type FakePodControl struct {
	controllerSpec []api.ReplicationController
	deletePodName  []string
	lock           sync.Mutex
}

func init() {
	api.ForTesting_ReferencesAllowBlankSelfLinks = true
}

func (f *FakePodControl) createReplica(namespace string, spec api.ReplicationController) {
	f.lock.Lock()
	defer f.lock.Unlock()
	f.controllerSpec = append(f.controllerSpec, spec)
}

func (f *FakePodControl) deletePod(namespace string, podName string) error {
	f.lock.Lock()
	defer f.lock.Unlock()
	f.deletePodName = append(f.deletePodName, podName)
	return nil
}

func newReplicationController(replicas int) api.ReplicationController {
	return api.ReplicationController{
		TypeMeta:   api.TypeMeta{APIVersion: testapi.Version()},
		ObjectMeta: api.ObjectMeta{Name: "foobar", Namespace: api.NamespaceDefault, ResourceVersion: "18"},
		Spec: api.ReplicationControllerSpec{
			Replicas: replicas,
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
}

func newPodList(count int) *api.PodList {
	pods := []api.Pod{}
	for i := 0; i < count; i++ {
		pods = append(pods, api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name: fmt.Sprintf("pod%d", i),
			},
		})
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
	if api.PreV1Beta3(testapi.Version()) {
		return "replicationControllers"
	}
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
	if !api.PreV1Beta3(testapi.Version()) && namespace != "" {
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

func TestSyncReplicationControllerDoesNothing(t *testing.T) {
	body, _ := latest.Codec.Encode(newPodList(2))
	fakeHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(body),
	}
	testServer := httptest.NewServer(&fakeHandler)
	defer testServer.Close()
	client := client.NewOrDie(&client.Config{Host: testServer.URL, Version: testapi.Version()})

	fakePodControl := FakePodControl{}

	manager := NewReplicationManager(client)
	manager.podControl = &fakePodControl

	controllerSpec := newReplicationController(2)

	manager.syncReplicationController(controllerSpec)
	validateSyncReplication(t, &fakePodControl, 0, 0)
}

func TestSyncReplicationControllerDeletes(t *testing.T) {
	body, _ := latest.Codec.Encode(newPodList(2))
	fakeHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(body),
	}
	testServer := httptest.NewServer(&fakeHandler)
	defer testServer.Close()
	client := client.NewOrDie(&client.Config{Host: testServer.URL, Version: testapi.Version()})

	fakePodControl := FakePodControl{}

	manager := NewReplicationManager(client)
	manager.podControl = &fakePodControl

	controllerSpec := newReplicationController(1)

	manager.syncReplicationController(controllerSpec)
	validateSyncReplication(t, &fakePodControl, 0, 1)
}

func TestSyncReplicationControllerCreates(t *testing.T) {
	controller := newReplicationController(2)
	testServer, fakeUpdateHandler := makeTestServer(t, api.NamespaceDefault, controller.Name,
		serverResponse{http.StatusOK, newPodList(0)},
		serverResponse{http.StatusInternalServerError, &api.ReplicationControllerList{}},
		serverResponse{http.StatusOK, &controller})
	defer testServer.Close()
	client := client.NewOrDie(&client.Config{Host: testServer.URL, Version: testapi.Version()})

	manager := NewReplicationManager(client)
	fakePodControl := FakePodControl{}
	manager.podControl = &fakePodControl
	manager.syncReplicationController(controller)
	validateSyncReplication(t, &fakePodControl, 2, 0)

	// No Status.Replicas update expected even though 2 pods were just created,
	// because the controller manager can't observe the pods till the next sync cycle.
	if fakeUpdateHandler.RequestReceived != nil {
		t.Errorf("Unexpected updates for controller via %v",
			fakeUpdateHandler.RequestReceived.URL)
	}
}

func TestCreateReplica(t *testing.T) {
	ns := api.NamespaceDefault
	body := runtime.EncodeOrDie(testapi.Codec(), &api.Pod{})
	fakeHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(body),
	}
	testServer := httptest.NewServer(&fakeHandler)
	defer testServer.Close()
	client := client.NewOrDie(&client.Config{Host: testServer.URL, Version: testapi.Version()})

	podControl := RealPodControl{
		kubeClient: client,
	}

	controllerSpec := newReplicationController(1)
	podControl.createReplica(ns, controllerSpec)

	manifest := api.ContainerManifest{}
	if err := api.Scheme.Convert(&controllerSpec.Spec.Template.Spec, &manifest); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expectedPod := api.Pod{
		ObjectMeta: api.ObjectMeta{
			Labels:       controllerSpec.Spec.Template.Labels,
			GenerateName: fmt.Sprintf("%s-", controllerSpec.Name),
		},
		Spec: controllerSpec.Spec.Template.Spec,
	}
	fakeHandler.ValidateRequest(t, testapi.ResourcePathWithQueryParams("pods", api.NamespaceDefault, ""), "POST", nil)
	actualPod, err := client.Codec.Decode([]byte(fakeHandler.RequestBody))
	if err != nil {
		t.Errorf("Unexpected error: %#v", err)
	}
	if !api.Semantic.DeepDerivative(&expectedPod, actualPod) {
		t.Logf("Body: %s", fakeHandler.RequestBody)
		t.Errorf("Unexpected mismatch.  Expected\n %#v,\n Got:\n %#v", &expectedPod, actualPod)
	}
}

func TestSynchronize(t *testing.T) {
	controllerSpec1 := newReplicationController(4)
	controllerSpec2 := newReplicationController(3)
	controllerSpec2.Name = "bar"
	controllerSpec2.Spec.Template.ObjectMeta.Labels = map[string]string{
		"name": "bar",
		"type": "production",
	}

	testServer, _ := makeTestServer(t, api.NamespaceDefault, "",
		serverResponse{http.StatusOK, newPodList(0)},
		serverResponse{http.StatusOK, &api.ReplicationControllerList{
			Items: []api.ReplicationController{
				controllerSpec1,
				controllerSpec2,
			}}},
		serverResponse{http.StatusInternalServerError, &api.ReplicationController{}})
	defer testServer.Close()
	client := client.NewOrDie(&client.Config{Host: testServer.URL, Version: testapi.Version()})
	manager := NewReplicationManager(client)
	fakePodControl := FakePodControl{}
	manager.podControl = &fakePodControl

	manager.synchronize()

	validateSyncReplication(t, &fakePodControl, 7, 0)
}

func TestControllerNoReplicaUpdate(t *testing.T) {
	// Steady state for the replication controller, no Status.Replicas updates expected
	rc := newReplicationController(5)
	rc.Status = api.ReplicationControllerStatus{Replicas: 5}
	activePods := 5

	testServer, fakeUpdateHandler := makeTestServer(t, api.NamespaceDefault, rc.Name,
		serverResponse{http.StatusOK, newPodList(activePods)},
		serverResponse{http.StatusOK, &api.ReplicationControllerList{
			Items: []api.ReplicationController{rc},
		}},
		serverResponse{http.StatusOK, &rc})
	defer testServer.Close()
	client := client.NewOrDie(&client.Config{Host: testServer.URL, Version: testapi.Version()})
	manager := NewReplicationManager(client)
	fakePodControl := FakePodControl{}
	manager.podControl = &fakePodControl

	manager.synchronize()

	validateSyncReplication(t, &fakePodControl, 0, 0)
	if fakeUpdateHandler.RequestReceived != nil {
		t.Errorf("Unexpected updates for controller via %v",
			fakeUpdateHandler.RequestReceived.URL)
	}
}

func TestControllerUpdateReplicas(t *testing.T) {
	// Insufficient number of pods in the system, and Status.Replicas is wrong;
	// Status.Replica should update to match number of pods in system, 1 new pod should be created.
	rc := newReplicationController(5)
	rc.Status = api.ReplicationControllerStatus{Replicas: 2}
	activePods := 4

	testServer, fakeUpdateHandler := makeTestServer(t, api.NamespaceDefault, rc.Name,
		serverResponse{http.StatusOK, newPodList(activePods)},
		serverResponse{http.StatusOK, &api.ReplicationControllerList{
			Items: []api.ReplicationController{rc},
		}},
		serverResponse{http.StatusOK, &rc})
	defer testServer.Close()
	client := client.NewOrDie(&client.Config{Host: testServer.URL, Version: testapi.Version()})
	manager := NewReplicationManager(client)
	fakePodControl := FakePodControl{}
	manager.podControl = &fakePodControl

	manager.synchronize()

	// Status.Replicas should go up from 2->4 even though we created 5-4=1 pod
	rc.Status = api.ReplicationControllerStatus{Replicas: 4}
	// These are set by default.
	rc.Spec.Selector = rc.Spec.Template.Labels
	rc.Labels = rc.Spec.Template.Labels

	decRc := runtime.EncodeOrDie(testapi.Codec(), &rc)
	fakeUpdateHandler.ValidateRequest(t, testapi.ResourcePathWithQueryParams(replicationControllerResourceName(), rc.Namespace, rc.Name), "PUT", &decRc)
	validateSyncReplication(t, &fakePodControl, 1, 0)
}

type FakeWatcher struct {
	w *watch.FakeWatcher
	*testclient.Fake
}

func TestWatchControllers(t *testing.T) {
	fakeWatch := watch.NewFake()
	client := &testclient.Fake{Watch: fakeWatch}
	manager := NewReplicationManager(client)
	var testControllerSpec api.ReplicationController
	received := make(chan struct{})
	manager.syncHandler = func(controllerSpec api.ReplicationController) error {
		if !api.Semantic.DeepDerivative(controllerSpec, testControllerSpec) {
			t.Errorf("Expected %#v, but got %#v", testControllerSpec, controllerSpec)
		}
		close(received)
		return nil
	}

	resourceVersion := ""
	go manager.watchControllers(&resourceVersion)

	// Test normal case
	testControllerSpec.Name = "foo"

	fakeWatch.Add(&testControllerSpec)

	select {
	case <-received:
	case <-time.After(100 * time.Millisecond):
		t.Errorf("Expected 1 call but got 0")
	}
}

func TestActivePodFiltering(t *testing.T) {
	podList := newPodList(5)
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
	podList := newPodList(numPods)
	pods := make([]*api.Pod, len(podList.Items))
	for i := range podList.Items {
		pods[i] = &podList.Items[i]
	}
	// pods[0] is not scheduled yet.
	pods[0].Spec.Host = ""
	pods[0].Status.Phase = api.PodPending
	// pods[1] is scheduled but pending.
	pods[1].Spec.Host = "bar"
	pods[1].Status.Phase = api.PodPending
	// pods[2] is unknown.
	pods[2].Spec.Host = "foo"
	pods[2].Status.Phase = api.PodUnknown
	// pods[3] is running but not ready.
	pods[3].Spec.Host = "foo"
	pods[3].Status.Phase = api.PodRunning
	// pods[4] is running and ready.
	pods[4].Spec.Host = "foo"
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
