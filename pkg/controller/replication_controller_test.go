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
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
	"github.com/coreos/go-etcd/etcd"
)

// TODO: Move this to a common place, it's needed in multiple tests.
var apiPath = "/api/v1beta1"

func makeURL(suffix string) string {
	return apiPath + suffix
}

type FakePodControl struct {
	controllerSpec []api.ReplicationController
	deletePodID    []string
	lock           sync.Mutex
}

func (f *FakePodControl) createReplica(spec api.ReplicationController) {
	f.lock.Lock()
	defer f.lock.Unlock()
	f.controllerSpec = append(f.controllerSpec, spec)
}

func (f *FakePodControl) deletePod(podID string) error {
	f.lock.Lock()
	defer f.lock.Unlock()
	f.deletePodID = append(f.deletePodID, podID)
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

func newPodList(count int) api.PodList {
	pods := []api.Pod{}
	for i := 0; i < count; i++ {
		pods = append(pods, api.Pod{
			JSONBase: api.JSONBase{
				ID: fmt.Sprintf("pod%d", i),
			},
		})
	}
	return api.PodList{
		Items: pods,
	}
}

func validateSyncReplication(t *testing.T, fakePodControl *FakePodControl, expectedCreates, expectedDeletes int) {
	if len(fakePodControl.controllerSpec) != expectedCreates {
		t.Errorf("Unexpected number of creates.  Expected %d, saw %d\n", expectedCreates, len(fakePodControl.controllerSpec))
	}
	if len(fakePodControl.deletePodID) != expectedDeletes {
		t.Errorf("Unexpected number of deletes.  Expected %d, saw %d\n", expectedDeletes, len(fakePodControl.deletePodID))
	}
}

func TestSyncReplicationControllerDoesNothing(t *testing.T) {
	body, _ := api.Encode(newPodList(2))
	fakeHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(body),
	}
	testServer := httptest.NewTLSServer(&fakeHandler)
	client := client.NewOrDie(testServer.URL, nil)

	fakePodControl := FakePodControl{}

	manager := NewReplicationManager(client)
	manager.podControl = &fakePodControl

	controllerSpec := newReplicationController(2)

	manager.syncReplicationController(controllerSpec)
	validateSyncReplication(t, &fakePodControl, 0, 0)
}

func TestSyncReplicationControllerDeletes(t *testing.T) {
	body, _ := api.Encode(newPodList(2))
	fakeHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(body),
	}
	testServer := httptest.NewTLSServer(&fakeHandler)
	client := client.NewOrDie(testServer.URL, nil)

	fakePodControl := FakePodControl{}

	manager := NewReplicationManager(client)
	manager.podControl = &fakePodControl

	controllerSpec := newReplicationController(1)

	manager.syncReplicationController(controllerSpec)
	validateSyncReplication(t, &fakePodControl, 0, 1)
}

func TestSyncReplicationControllerCreates(t *testing.T) {
	body, _ := api.Encode(newPodList(0))
	fakeHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(body),
	}
	testServer := httptest.NewTLSServer(&fakeHandler)
	client := client.NewOrDie(testServer.URL, nil)

	fakePodControl := FakePodControl{}

	manager := NewReplicationManager(client)
	manager.podControl = &fakePodControl

	controllerSpec := newReplicationController(2)

	manager.syncReplicationController(controllerSpec)
	validateSyncReplication(t, &fakePodControl, 2, 0)
}

func TestCreateReplica(t *testing.T) {
	body, _ := api.Encode(api.Pod{})
	fakeHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(body),
	}
	testServer := httptest.NewTLSServer(&fakeHandler)
	client := client.NewOrDie(testServer.URL, nil)

	podControl := RealPodControl{
		kubeClient: client,
	}

	controllerSpec := api.ReplicationController{
		JSONBase: api.JSONBase{
			Kind: "ReplicationController",
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
					"name": "foo",
					"type": "production",
				},
			},
		},
	}

	podControl.createReplica(controllerSpec)

	expectedPod := api.Pod{
		JSONBase: api.JSONBase{
			Kind:       "Pod",
			APIVersion: "v1beta1",
		},
		Labels:       controllerSpec.DesiredState.PodTemplate.Labels,
		DesiredState: controllerSpec.DesiredState.PodTemplate.DesiredState,
	}
	fakeHandler.ValidateRequest(t, makeURL("/pods"), "POST", nil)
	actualPod := api.Pod{}
	if err := json.Unmarshal([]byte(fakeHandler.RequestBody), &actualPod); err != nil {
		t.Errorf("Unexpected error: %#v", err)
	}
	if !reflect.DeepEqual(expectedPod, actualPod) {
		t.Logf("Body: %s", fakeHandler.RequestBody)
		t.Errorf("Unexpected mismatch.  Expected\n %#v,\n Got:\n %#v", expectedPod, actualPod)
	}
}

func TestSyncronize(t *testing.T) {
	controllerSpec1 := api.ReplicationController{
		JSONBase: api.JSONBase{APIVersion: "v1beta1"},
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
		JSONBase: api.JSONBase{APIVersion: "v1beta1"},
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
		ResponseBody: "{\"apiVersion\": \"v1beta1\", \"kind\": \"PodList\"}",
		T:            t,
	}
	fakeControllerHandler := util.FakeHandler{
		StatusCode: 200,
		ResponseBody: api.EncodeOrDie(&api.ReplicationControllerList{
			Items: []api.ReplicationController{
				controllerSpec1,
				controllerSpec2,
			},
		}),
		T: t,
	}
	mux := http.NewServeMux()
	mux.Handle("/api/v1beta1/pods/", &fakePodHandler)
	mux.Handle("/api/v1beta1/replicationControllers/", &fakeControllerHandler)
	mux.HandleFunc("/", func(w http.ResponseWriter, req *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		t.Errorf("Unexpected request for %v", req.RequestURI)
	})
	testServer := httptest.NewServer(mux)
	client := client.NewOrDie(testServer.URL, nil)
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

func (fw FakeWatcher) WatchReplicationControllers(l, f labels.Selector, rv uint64) (watch.Interface, error) {
	return fw.w, nil
}

func TestWatchControllers(t *testing.T) {
	client := FakeWatcher{watch.NewFake(), &client.Fake{}}
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

	resourceVersion := uint64(0)
	go manager.watchControllers(&resourceVersion)

	// Test normal case
	testControllerSpec.ID = "foo"
	client.w.Add(&testControllerSpec)

	select {
	case <-received:
	case <-time.After(10 * time.Millisecond):
		t.Errorf("Expected 1 call but got 0")
	}
}
