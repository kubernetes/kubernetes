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

package service

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	_ "github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

func newPodList(count int) api.PodList {
	pods := []api.Pod{}
	for i := 0; i < count; i++ {
		pods = append(pods, api.Pod{
			TypeMeta:   api.TypeMeta{APIVersion: testapi.Version()},
			ObjectMeta: api.ObjectMeta{Name: fmt.Sprintf("pod%d", i)},
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
		TypeMeta: api.TypeMeta{APIVersion: testapi.Version(), Kind: "PodList"},
		Items:    pods,
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
	emptyPortsManifest := api.ContainerManifest{
		Containers: []api.Container{
			{
				Ports: []api.Port{},
			},
		},
	}
	tests := []struct {
		manifest api.ContainerManifest
		portName util.IntOrString

		wport int
		werr  bool
	}{
		{
			manifest,
			util.IntOrString{Kind: util.IntstrString, StrVal: "foo"},
			8080,
			false,
		},
		{
			manifest,
			util.IntOrString{Kind: util.IntstrString, StrVal: "bar"},
			8000,
			false,
		},
		{
			manifest,
			util.IntOrString{Kind: util.IntstrInt, IntVal: 8000},
			8000,
			false,
		},
		{
			manifest,
			util.IntOrString{Kind: util.IntstrInt, IntVal: 7000},
			7000,
			false,
		},
		{
			manifest,
			util.IntOrString{Kind: util.IntstrString, StrVal: "baz"},
			0,
			true,
		},
		{
			manifest,
			util.IntOrString{Kind: util.IntstrString, StrVal: ""},
			8080,
			false,
		},
		{
			manifest,
			util.IntOrString{Kind: util.IntstrInt, IntVal: 0},
			8080,
			false,
		},
		{
			emptyPortsManifest,
			util.IntOrString{Kind: util.IntstrString, StrVal: ""},
			0,
			true,
		},
		{
			emptyPortsManifest,
			util.IntOrString{Kind: util.IntstrInt, IntVal: 0},
			0,
			true,
		},
	}
	for _, test := range tests {
		port, err := findPort(&test.manifest, test.portName)
		if port != test.wport {
			t.Errorf("Expected port %d, Got %d", test.wport, port)
		}
		if err == nil && test.werr {
			t.Errorf("unexpected non-error")
		}
		if err != nil && test.werr == false {
			t.Errorf("unexpected error: %v", err)
		}
	}
}

type serverResponse struct {
	statusCode int
	obj        interface{}
}

func makeTestServer(t *testing.T, podResponse serverResponse, serviceResponse serverResponse, endpointsResponse serverResponse) (*httptest.Server, *util.FakeHandler) {
	fakePodHandler := util.FakeHandler{
		StatusCode:   podResponse.statusCode,
		ResponseBody: util.EncodeJSON(podResponse.obj),
	}
	fakeServiceHandler := util.FakeHandler{
		StatusCode:   serviceResponse.statusCode,
		ResponseBody: util.EncodeJSON(serviceResponse.obj),
	}
	fakeEndpointsHandler := util.FakeHandler{
		StatusCode:   endpointsResponse.statusCode,
		ResponseBody: util.EncodeJSON(endpointsResponse.obj),
	}
	mux := http.NewServeMux()
	mux.Handle("/api/"+testapi.Version()+"/pods", &fakePodHandler)
	mux.Handle("/api/"+testapi.Version()+"/services", &fakeServiceHandler)
	mux.Handle("/api/"+testapi.Version()+"/endpoints", &fakeEndpointsHandler)
	mux.Handle("/api/"+testapi.Version()+"/endpoints/", &fakeEndpointsHandler)
	mux.HandleFunc("/", func(res http.ResponseWriter, req *http.Request) {
		t.Errorf("unexpected request: %v", req.RequestURI)
		res.WriteHeader(http.StatusNotFound)
	})
	return httptest.NewServer(mux), &fakeEndpointsHandler
}

func TestSyncEndpointsEmpty(t *testing.T) {
	testServer, _ := makeTestServer(t,
		serverResponse{http.StatusOK, newPodList(0)},
		serverResponse{http.StatusOK, api.ServiceList{}},
		serverResponse{http.StatusOK, api.Endpoints{}})
	defer testServer.Close()
	client := client.NewOrDie(&client.Config{Host: testServer.URL, Version: testapi.Version()})
	endpoints := NewEndpointController(client)
	if err := endpoints.SyncServiceEndpoints(); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestSyncEndpointsError(t *testing.T) {
	testServer, _ := makeTestServer(t,
		serverResponse{http.StatusOK, newPodList(0)},
		serverResponse{http.StatusInternalServerError, api.ServiceList{}},
		serverResponse{http.StatusOK, api.Endpoints{}})
	defer testServer.Close()
	client := client.NewOrDie(&client.Config{Host: testServer.URL, Version: testapi.Version()})
	endpoints := NewEndpointController(client)
	if err := endpoints.SyncServiceEndpoints(); err == nil {
		t.Errorf("unexpected non-error")
	}
}

func TestSyncEndpointsItemsPreexisting(t *testing.T) {
	serviceList := api.ServiceList{
		Items: []api.Service{
			{
				ObjectMeta: api.ObjectMeta{Name: "foo"},
				Spec: api.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
					},
				},
			},
		},
	}
	testServer, endpointsHandler := makeTestServer(t,
		serverResponse{http.StatusOK, newPodList(1)},
		serverResponse{http.StatusOK, serviceList},
		serverResponse{http.StatusOK, api.Endpoints{
			ObjectMeta: api.ObjectMeta{
				Name:            "foo",
				ResourceVersion: "1",
			},
			Endpoints: []string{"6.7.8.9:1000"},
		}})
	defer testServer.Close()
	client := client.NewOrDie(&client.Config{Host: testServer.URL, Version: testapi.Version()})
	endpoints := NewEndpointController(client)
	if err := endpoints.SyncServiceEndpoints(); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	data := runtime.EncodeOrDie(testapi.Codec(), &api.Endpoints{
		ObjectMeta: api.ObjectMeta{
			Name:            "foo",
			ResourceVersion: "1",
		},
		Endpoints: []string{"1.2.3.4:8080"},
	})
	endpointsHandler.ValidateRequest(t, "/api/"+testapi.Version()+"/endpoints/foo", "PUT", &data)
}

func TestSyncEndpointsItemsPreexistingIdentical(t *testing.T) {
	serviceList := api.ServiceList{
		Items: []api.Service{
			{
				ObjectMeta: api.ObjectMeta{Name: "foo"},
				Spec: api.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
					},
				},
			},
		},
	}
	testServer, endpointsHandler := makeTestServer(t,
		serverResponse{http.StatusOK, newPodList(1)},
		serverResponse{http.StatusOK, serviceList},
		serverResponse{http.StatusOK, api.Endpoints{
			ObjectMeta: api.ObjectMeta{
				ResourceVersion: "1",
			},
			Endpoints: []string{"1.2.3.4:8080"},
		}})
	defer testServer.Close()
	client := client.NewOrDie(&client.Config{Host: testServer.URL, Version: testapi.Version()})
	endpoints := NewEndpointController(client)
	if err := endpoints.SyncServiceEndpoints(); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	endpointsHandler.ValidateRequest(t, "/api/"+testapi.Version()+"/endpoints/foo", "GET", nil)
}

func TestSyncEndpointsItems(t *testing.T) {
	serviceList := api.ServiceList{
		Items: []api.Service{
			{
				ObjectMeta: api.ObjectMeta{Name: "foo"},
				Spec: api.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
					},
				},
			},
		},
	}
	testServer, endpointsHandler := makeTestServer(t,
		serverResponse{http.StatusOK, newPodList(1)},
		serverResponse{http.StatusOK, serviceList},
		serverResponse{http.StatusOK, api.Endpoints{}})
	defer testServer.Close()
	client := client.NewOrDie(&client.Config{Host: testServer.URL, Version: testapi.Version()})
	endpoints := NewEndpointController(client)
	if err := endpoints.SyncServiceEndpoints(); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	data := runtime.EncodeOrDie(testapi.Codec(), &api.Endpoints{
		ObjectMeta: api.ObjectMeta{
			ResourceVersion: "",
		},
		Endpoints: []string{"1.2.3.4:8080"},
	})
	endpointsHandler.ValidateRequest(t, "/api/"+testapi.Version()+"/endpoints", "POST", &data)
}

func TestSyncEndpointsPodError(t *testing.T) {
	serviceList := api.ServiceList{
		Items: []api.Service{
			{
				Spec: api.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
					},
				},
			},
		},
	}
	testServer, _ := makeTestServer(t,
		serverResponse{http.StatusInternalServerError, api.PodList{}},
		serverResponse{http.StatusOK, serviceList},
		serverResponse{http.StatusOK, api.Endpoints{}})
	defer testServer.Close()
	client := client.NewOrDie(&client.Config{Host: testServer.URL, Version: testapi.Version()})
	endpoints := NewEndpointController(client)
	if err := endpoints.SyncServiceEndpoints(); err == nil {
		t.Error("Unexpected non-error")
	}
}
