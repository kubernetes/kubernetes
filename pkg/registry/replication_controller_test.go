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
package registry

import (
	"encoding/json"
	"fmt"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/coreos/go-etcd/etcd"
)

// TODO: Move this to a common place, it's needed in multiple tests.
var apiPath = "/api/v1beta1"

func makeUrl(suffix string) string {
	return apiPath + suffix
}

type FakePodControl struct {
	controllerSpec []api.ReplicationController
	deletePodID    []string
}

func (f *FakePodControl) createReplica(spec api.ReplicationController) {
	f.controllerSpec = append(f.controllerSpec, spec)
}

func (f *FakePodControl) deletePod(podID string) error {
	f.deletePodID = append(f.deletePodID, podID)
	return nil
}

func makeReplicationController(replicas int) api.ReplicationController {
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

func makePodList(count int) api.PodList {
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
	body, _ := json.Marshal(makePodList(2))
	fakeHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(body),
	}
	testServer := httptest.NewTLSServer(&fakeHandler)
	client := client.Client{
		Host: testServer.URL,
	}

	fakePodControl := FakePodControl{}

	manager := MakeReplicationManager(nil, &client)
	manager.podControl = &fakePodControl

	controllerSpec := makeReplicationController(2)

	manager.syncReplicationController(controllerSpec)
	validateSyncReplication(t, &fakePodControl, 0, 0)
}

func TestSyncReplicationControllerDeletes(t *testing.T) {
	body, _ := json.Marshal(makePodList(2))
	fakeHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(body),
	}
	testServer := httptest.NewTLSServer(&fakeHandler)
	client := client.Client{
		Host: testServer.URL,
	}

	fakePodControl := FakePodControl{}

	manager := MakeReplicationManager(nil, &client)
	manager.podControl = &fakePodControl

	controllerSpec := makeReplicationController(1)

	manager.syncReplicationController(controllerSpec)
	validateSyncReplication(t, &fakePodControl, 0, 1)
}

func TestSyncReplicationControllerCreates(t *testing.T) {
	body := "{ \"items\": [] }"
	fakeHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(body),
	}
	testServer := httptest.NewTLSServer(&fakeHandler)
	client := client.Client{
		Host: testServer.URL,
	}

	fakePodControl := FakePodControl{}

	manager := MakeReplicationManager(nil, &client)
	manager.podControl = &fakePodControl

	controllerSpec := makeReplicationController(2)

	manager.syncReplicationController(controllerSpec)
	validateSyncReplication(t, &fakePodControl, 2, 0)
}

func TestCreateReplica(t *testing.T) {
	body := "{}"
	fakeHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(body),
	}
	testServer := httptest.NewTLSServer(&fakeHandler)
	client := client.Client{
		Host: testServer.URL,
	}

	podControl := RealPodControl{
		kubeClient: client,
	}

	controllerSpec := api.ReplicationController{
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

	//expectedPod := Pod{
	//	Labels:       controllerSpec.DesiredState.PodTemplate.Labels,
	//	DesiredState: controllerSpec.DesiredState.PodTemplate.DesiredState,
	//}
	// TODO: fix this so that it validates the body.
	fakeHandler.ValidateRequest(t, makeUrl("/pods"), "POST", nil)
}

func TestHandleWatchResponseNotSet(t *testing.T) {
	body, _ := json.Marshal(makePodList(2))
	fakeHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(body),
	}
	testServer := httptest.NewTLSServer(&fakeHandler)
	client := client.Client{
		Host: testServer.URL,
	}

	fakePodControl := FakePodControl{}

	manager := MakeReplicationManager(nil, &client)
	manager.podControl = &fakePodControl
	_, err := manager.handleWatchResponse(&etcd.Response{
		Action: "delete",
	})
	expectNoError(t, err)
}

func TestHandleWatchResponseNoNode(t *testing.T) {
	body, _ := json.Marshal(makePodList(2))
	fakeHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(body),
	}
	testServer := httptest.NewTLSServer(&fakeHandler)
	client := client.Client{
		Host: testServer.URL,
	}

	fakePodControl := FakePodControl{}

	manager := MakeReplicationManager(nil, &client)
	manager.podControl = &fakePodControl
	_, err := manager.handleWatchResponse(&etcd.Response{
		Action: "set",
	})
	if err == nil {
		t.Error("Unexpected non-error")
	}
}

func TestHandleWatchResponseBadData(t *testing.T) {
	body, _ := json.Marshal(makePodList(2))
	fakeHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(body),
	}
	testServer := httptest.NewTLSServer(&fakeHandler)
	client := client.Client{
		Host: testServer.URL,
	}

	fakePodControl := FakePodControl{}

	manager := MakeReplicationManager(nil, &client)
	manager.podControl = &fakePodControl
	_, err := manager.handleWatchResponse(&etcd.Response{
		Action: "set",
		Node: &etcd.Node{
			Value: "foobar",
		},
	})
	if err == nil {
		t.Error("Unexpected non-error")
	}
}

func TestHandleWatchResponse(t *testing.T) {
	body, _ := json.Marshal(makePodList(2))
	fakeHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(body),
	}
	testServer := httptest.NewTLSServer(&fakeHandler)
	client := client.Client{
		Host: testServer.URL,
	}

	fakePodControl := FakePodControl{}

	manager := MakeReplicationManager(nil, &client)
	manager.podControl = &fakePodControl

	controller := makeReplicationController(2)

	data, err := json.Marshal(controller)
	expectNoError(t, err)
	controllerOut, err := manager.handleWatchResponse(&etcd.Response{
		Action: "set",
		Node: &etcd.Node{
			Value: string(data),
		},
	})
	if err != nil {
		t.Errorf("Unexpected error: %#v", err)
	}
	if !reflect.DeepEqual(controller, *controllerOut) {
		t.Errorf("Unexpected mismatch.  Expected %#v, Saw: %#v", controller, controllerOut)
	}
}
