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
	"net/http"
	"net/http/httptest"
	"path"
	"sync"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

func makeNamespaceURL(namespace, suffix string) string {
	if !(testapi.Version() == "v1beta1" || testapi.Version() == "v1beta2") {
		return makeURL("/ns/" + namespace + suffix)
	}
	return makeURL(suffix + "?namespace=" + namespace)
}

func makeURL(suffix string) string {
	return path.Join("/api", testapi.Version(), suffix)
}

type FakePodControl struct {
	controllerSpec []api.ReplicationController
	deletePodName  []string
	lock           sync.Mutex
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
		ObjectMeta: api.ObjectMeta{Name: "foobar", Namespace: "default", ResourceVersion: "18"},
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
	body := runtime.EncodeOrDie(testapi.Codec(), newPodList(0))
	fakePodHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(body),
	}
	fakePodControl := FakePodControl{}

	controller := newReplicationController(2)
	fakeUpdateHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: runtime.EncodeOrDie(testapi.Codec(), &controller),
		T:            t,
	}

	testServerMux := http.NewServeMux()
	testServerMux.Handle("/api/"+testapi.Version()+"/pods/", &fakePodHandler)
	testServerMux.Handle(fmt.Sprintf("/api/"+testapi.Version()+"/replicationControllers/%s", controller.Name), &fakeUpdateHandler)
	testServer := httptest.NewServer(testServerMux)
	defer testServer.Close()
	client := client.NewOrDie(&client.Config{Host: testServer.URL, Version: testapi.Version()})

	manager := NewReplicationManager(client)
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
	fakeHandler.ValidateRequest(t, makeNamespaceURL("default", "/pods"), "POST", nil)
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

	fakePodHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: "{\"apiVersion\": \"" + testapi.Version() + "\", \"kind\": \"PodList\"}",
		T:            t,
	}
	fakeControllerHandler := util.FakeHandler{
		StatusCode: 200,
		ResponseBody: runtime.EncodeOrDie(latest.Codec, &api.ReplicationControllerList{
			Items: []api.ReplicationController{
				controllerSpec1,
				controllerSpec2,
			},
		}),
		T: t,
	}
	mux := http.NewServeMux()
	mux.Handle("/api/"+testapi.Version()+"/pods/", &fakePodHandler)
	mux.Handle("/api/"+testapi.Version()+"/replicationControllers/", &fakeControllerHandler)
	mux.HandleFunc("/", func(w http.ResponseWriter, req *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		t.Errorf("Unexpected request for %v", req.RequestURI)
	})
	testServer := httptest.NewServer(mux)
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

	body, _ := latest.Codec.Encode(newPodList(activePods))
	fakePodHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(body),
		T:            t,
	}
	fakeControllerHandler := util.FakeHandler{
		StatusCode: 200,
		ResponseBody: runtime.EncodeOrDie(latest.Codec, &api.ReplicationControllerList{
			Items: []api.ReplicationController{rc},
		}),
		T: t,
	}
	fakeUpdateHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: runtime.EncodeOrDie(testapi.Codec(), &rc),
		T:            t,
	}

	mux := http.NewServeMux()
	mux.Handle("/api/"+testapi.Version()+"/pods/", &fakePodHandler)
	mux.Handle("/api/"+testapi.Version()+"/replicationControllers/", &fakeControllerHandler)
	mux.Handle(fmt.Sprintf("/api/"+testapi.Version()+"/replicationControllers/%s", rc.Name), &fakeUpdateHandler)
	mux.HandleFunc("/", func(w http.ResponseWriter, req *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		t.Errorf("Unexpected request for %v", req.RequestURI)
	})
	testServer := httptest.NewServer(mux)
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

	body, _ := latest.Codec.Encode(newPodList(activePods))
	fakePodHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(body),
		T:            t,
	}
	fakeControllerHandler := util.FakeHandler{
		StatusCode: 200,
		ResponseBody: runtime.EncodeOrDie(latest.Codec, &api.ReplicationControllerList{
			Items: []api.ReplicationController{rc},
		}),
		T: t,
	}
	fakeUpdateHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: runtime.EncodeOrDie(testapi.Codec(), &rc),
		T:            t,
	}

	mux := http.NewServeMux()

	mux.Handle("/api/"+testapi.Version()+"/pods/", &fakePodHandler)
	mux.Handle("/api/"+testapi.Version()+"/replicationControllers/", &fakeControllerHandler)
	mux.Handle(fmt.Sprintf("/api/"+testapi.Version()+"/replicationControllers/%s", rc.Name), &fakeUpdateHandler)
	mux.HandleFunc("/", func(w http.ResponseWriter, req *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		t.Errorf("Unexpected request for %v", req.RequestURI)
	})
	testServer := httptest.NewServer(mux)
	defer testServer.Close()
	client := client.NewOrDie(&client.Config{Host: testServer.URL, Version: testapi.Version()})
	manager := NewReplicationManager(client)
	fakePodControl := FakePodControl{}
	manager.podControl = &fakePodControl

	manager.synchronize()

	// Status.Replicas should go up from 2->4 even though we created 5-4=1 pod
	rc.Status = api.ReplicationControllerStatus{Replicas: 4}
	decRc := runtime.EncodeOrDie(testapi.Codec(), &rc)
	fakeUpdateHandler.ValidateRequest(t, fmt.Sprintf("/api/"+testapi.Version()+"/replicationControllers/%s?namespace=%s", rc.Name, rc.Namespace), "PUT", &decRc)
	validateSyncReplication(t, &fakePodControl, 1, 0)
}

type FakeWatcher struct {
	w *watch.FakeWatcher
	*client.Fake
}

func TestWatchControllers(t *testing.T) {
	fakeWatch := watch.NewFake()
	client := &client.Fake{Watch: fakeWatch}
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
	case <-time.After(10 * time.Millisecond):
		t.Errorf("Expected 1 call but got 0")
	}
}
