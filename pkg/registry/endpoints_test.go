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
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

func makePodList(count int) api.PodList {
	pods := []api.Pod{}
	for i := 0; i < count; i++ {
		pods = append(pods, api.Pod{
			JSONBase: api.JSONBase{
				ID: fmt.Sprintf("pod%d", i),
			},
			DesiredState: api.PodState{
				Manifest: api.ContainerManifest{
					Containers: []api.Container{
						{
							Ports: []api.Port{
								{
									ContainerPort: 8080,
								},
							},
						},
					},
				},
			},
			CurrentState: api.PodState{
				PodIP: "1.2.3.4",
			},
		})
	}
	return api.PodList{
		Items: pods,
	}
}

func TestFindPort(t *testing.T) {
	manifest := api.ContainerManifest{
		Containers: []api.Container{
			{
				Ports: []api.Port{
					{
						Name:          "foo",
						ContainerPort: 8080,
						HostPort:      9090,
					},
					{
						Name:          "bar",
						ContainerPort: 8000,
						HostPort:      9000,
					},
				},
			},
		},
	}
	port, err := findPort(&manifest, util.IntOrString{Kind: util.IntstrString, StrVal: "foo"})
	expectNoError(t, err)
	if port != 8080 {
		t.Errorf("Expected 8080, Got %d", port)
	}
	port, err = findPort(&manifest, util.IntOrString{Kind: util.IntstrString, StrVal: "bar"})
	expectNoError(t, err)
	if port != 8000 {
		t.Errorf("Expected 8000, Got %d", port)
	}
	port, err = findPort(&manifest, util.IntOrString{Kind: util.IntstrInt, IntVal: 8000})
	if port != 8000 {
		t.Errorf("Expected 8000, Got %d", port)
	}
	port, err = findPort(&manifest, util.IntOrString{Kind: util.IntstrInt, IntVal: 7000})
	if port != 7000 {
		t.Errorf("Expected 7000, Got %d", port)
	}
	port, err = findPort(&manifest, util.IntOrString{Kind: util.IntstrString, StrVal: "baz"})
	if err == nil {
		t.Error("unexpected non-error")
	}
	port, err = findPort(&manifest, util.IntOrString{Kind: util.IntstrString, StrVal: ""})
	expectNoError(t, err)
	if port != 8080 {
		t.Errorf("Expected 8080, Got %d", port)
	}
	port, err = findPort(&manifest, util.IntOrString{})
	expectNoError(t, err)
	if port != 8080 {
		t.Errorf("Expected 8080, Got %d", port)
	}
}

func TestSyncEndpointsEmpty(t *testing.T) {
	body, _ := json.Marshal(makePodList(0))
	fakeHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(body),
	}
	testServer := httptest.NewTLSServer(&fakeHandler)
	client := client.New(testServer.URL, nil)

	serviceRegistry := MockServiceRegistry{}

	endpoints := MakeEndpointController(&serviceRegistry, client)
	err := endpoints.SyncServiceEndpoints()
	expectNoError(t, err)
}

func TestSyncEndpointsError(t *testing.T) {
	body, _ := json.Marshal(makePodList(0))
	fakeHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(body),
	}
	testServer := httptest.NewTLSServer(&fakeHandler)
	client := client.New(testServer.URL, nil)

	serviceRegistry := MockServiceRegistry{
		err: fmt.Errorf("test error"),
	}

	endpoints := MakeEndpointController(&serviceRegistry, client)
	err := endpoints.SyncServiceEndpoints()
	if err != serviceRegistry.err {
		t.Errorf("Errors don't match: %#v %#v", err, serviceRegistry.err)
	}
}

func TestSyncEndpointsItems(t *testing.T) {
	body, _ := json.Marshal(makePodList(1))
	fakeHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(body),
	}
	testServer := httptest.NewTLSServer(&fakeHandler)
	client := client.New(testServer.URL, nil)

	serviceRegistry := MockServiceRegistry{
		list: api.ServiceList{
			Items: []api.Service{
				{
					Selector: map[string]string{
						"foo": "bar",
					},
				},
			},
		},
	}

	endpoints := MakeEndpointController(&serviceRegistry, client)
	err := endpoints.SyncServiceEndpoints()
	expectNoError(t, err)
	if len(serviceRegistry.endpoints.Endpoints) != 1 {
		t.Errorf("Unexpected endpoints update: %#v", serviceRegistry.endpoints)
	}
}

func TestSyncEndpointsPodError(t *testing.T) {
	fakeHandler := util.FakeHandler{
		StatusCode: 500,
	}
	testServer := httptest.NewTLSServer(&fakeHandler)
	client := client.New(testServer.URL, nil)

	serviceRegistry := MockServiceRegistry{
		list: api.ServiceList{
			Items: []api.Service{
				{
					Selector: map[string]string{
						"foo": "bar",
					},
				},
			},
		},
	}

	endpoints := MakeEndpointController(&serviceRegistry, client)
	err := endpoints.SyncServiceEndpoints()
	if err == nil {
		t.Error("Unexpected non-error")
	}
}
