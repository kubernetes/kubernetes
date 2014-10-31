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
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
	"github.com/coreos/go-etcd/etcd"
)

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
		DesiredState: api.ReplicationControllerState{
			Replicas: replicas,
			PodTemplate: api.PodTemplate{
				DesiredState: api.PodState{
					Manifest: api.ContainerManifest{
						Containers: []api.Container{
							{
								Image: "foo/bar",
							},
						},
					},
				},
				Labels: map[string]string{
					"name": "foo",
					"type": "production",
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
	validateSyncReplication(t, &fakePodControl, 2, 0)
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

	controllerSpec := api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name: "test",
		},
		DesiredState: api.ReplicationControllerState{
			PodTemplate: api.PodTemplate{
				DesiredState: api.PodState{
					Manifest: api.ContainerManifest{
						Containers: []api.Container{
							{
								Image: "foo/bar",
							},
						},
					},
				},
				Labels: map[string]string{
					"name":                  "foo",
					"type":                  "production",
					"replicationController": "test",
				},
			},
		},
	}

	podControl.createReplica(ns, controllerSpec)

	expectedPod := api.Pod{
		ObjectMeta: api.ObjectMeta{
			Labels: controllerSpec.DesiredState.PodTemplate.Labels,
		},
		DesiredState: controllerSpec.DesiredState.PodTemplate.DesiredState,
	}
	fakeHandler.ValidateRequest(t, makeURL("/pods?namespace=default"), "POST", nil)
	actualPod, err := client.Codec.Decode([]byte(fakeHandler.RequestBody))
	if err != nil {
		t.Errorf("Unexpected error: %#v", err)
	}
	if !reflect.DeepEqual(&expectedPod, actualPod) {
		t.Logf("Body: %s", fakeHandler.RequestBody)
		t.Errorf("Unexpected mismatch.  Expected\n %#v,\n Got:\n %#v", &expectedPod, actualPod)
	}
}

func TestSynchonize(t *testing.T) {
	controllerSpec1 := api.ReplicationController{
		TypeMeta: api.TypeMeta{APIVersion: testapi.Version()},
		DesiredState: api.ReplicationControllerState{
			Replicas: 4,
			PodTemplate: api.PodTemplate{
				DesiredState: api.PodState{
					Manifest: api.ContainerManifest{
						Containers: []api.Container{
							{
								Image: "foo/bar",
							},
						},
					},
				},
				Labels: map[string]string{
					"name": "foo",
					"type": "production",
				},
			},
		},
	}
	controllerSpec2 := api.ReplicationController{
		TypeMeta: api.TypeMeta{APIVersion: testapi.Version()},
		DesiredState: api.ReplicationControllerState{
			Replicas: 3,
			PodTemplate: api.PodTemplate{
				DesiredState: api.PodState{
					Manifest: api.ContainerManifest{
						Containers: []api.Container{
							{
								Image: "bar/baz",
							},
						},
					},
				},
				Labels: map[string]string{
					"name": "bar",
					"type": "production",
				},
			},
		},
	}

	fakeEtcd := tools.NewFakeEtcdClient(t)
	fakeEtcd.Data["/registry/controllers"] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Nodes: []*etcd.Node{
					{
						Value: util.EncodeJSON(controllerSpec1),
					},
					{
						Value: util.EncodeJSON(controllerSpec2),
					},
				},
			},
		},
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
		if !reflect.DeepEqual(controllerSpec, testControllerSpec) {
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
