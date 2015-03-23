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

func newPodList(count int) *api.PodList {
	pods := []api.Pod{}
	for i := 0; i < count; i++ {
		pods = append(pods, api.Pod{
			TypeMeta:   api.TypeMeta{APIVersion: testapi.Version()},
			ObjectMeta: api.ObjectMeta{Name: fmt.Sprintf("pod%d", i)},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Ports: []api.ContainerPort{
							{
								ContainerPort: 8080,
							},
						},
					},
				},
			},
			Status: api.PodStatus{
				PodIP: "1.2.3.4",
				Conditions: []api.PodCondition{
					{
						Type:   api.PodReady,
						Status: api.ConditionTrue,
					},
				},
			},
		})
	}
	return &api.PodList{
		TypeMeta: api.TypeMeta{APIVersion: testapi.Version(), Kind: "PodList"},
		Items:    pods,
	}
}

func TestFindPort(t *testing.T) {
	pod := api.Pod{
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Ports: []api.ContainerPort{
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
						{
							Name:          "default",
							ContainerPort: 8100,
							HostPort:      9200,
						},
					},
				},
			},
		},
	}

	emptyPortsPod := api.Pod{
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Ports: []api.ContainerPort{},
				},
			},
		},
	}

	singlePortPod := api.Pod{
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Ports: []api.ContainerPort{
						{
							ContainerPort: 8300,
						},
					},
				},
			},
		},
	}

	noDefaultPod := api.Pod{
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Ports: []api.ContainerPort{
						{
							Name:          "foo",
							ContainerPort: 8300,
						},
					},
				},
			},
		},
	}

	servicePort := 999

	tests := []struct {
		pod      api.Pod
		portName util.IntOrString

		wport int
		werr  bool
	}{
		{
			pod,
			util.IntOrString{Kind: util.IntstrString, StrVal: "foo"},
			8080,
			false,
		},
		{
			pod,
			util.IntOrString{Kind: util.IntstrString, StrVal: "bar"},
			8000,
			false,
		},
		{
			pod,
			util.IntOrString{Kind: util.IntstrInt, IntVal: 8000},
			8000,
			false,
		},
		{
			pod,
			util.IntOrString{Kind: util.IntstrInt, IntVal: 7000},
			7000,
			false,
		},
		{
			pod,
			util.IntOrString{Kind: util.IntstrString, StrVal: "baz"},
			0,
			true,
		},
		{
			emptyPortsPod,
			util.IntOrString{Kind: util.IntstrString, StrVal: "foo"},
			0,
			true,
		},
		{
			emptyPortsPod,
			util.IntOrString{Kind: util.IntstrString, StrVal: ""},
			servicePort,
			false,
		},
		{
			emptyPortsPod,
			util.IntOrString{Kind: util.IntstrInt, IntVal: 0},
			servicePort,
			false,
		},
		{
			singlePortPod,
			util.IntOrString{Kind: util.IntstrString, StrVal: ""},
			8300,
			false,
		},
		{
			singlePortPod,
			util.IntOrString{Kind: util.IntstrInt, IntVal: 0},
			8300,
			false,
		},
		{
			noDefaultPod,
			util.IntOrString{Kind: util.IntstrString, StrVal: ""},
			8300,
			false,
		},
		{
			noDefaultPod,
			util.IntOrString{Kind: util.IntstrInt, IntVal: 0},
			8300,
			false,
		},
	}
	for _, test := range tests {
		port, err := findPort(&test.pod, &api.Service{Spec: api.ServiceSpec{Port: servicePort, TargetPort: test.portName}})
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
		ResponseBody: runtime.EncodeOrDie(testapi.Codec(), podResponse.obj.(runtime.Object)),
	}
	fakeServiceHandler := util.FakeHandler{
		StatusCode:   serviceResponse.statusCode,
		ResponseBody: runtime.EncodeOrDie(testapi.Codec(), serviceResponse.obj.(runtime.Object)),
	}
	fakeEndpointsHandler := util.FakeHandler{
		StatusCode:   endpointsResponse.statusCode,
		ResponseBody: runtime.EncodeOrDie(testapi.Codec(), endpointsResponse.obj.(runtime.Object)),
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
		serverResponse{http.StatusOK, &api.ServiceList{}},
		serverResponse{http.StatusOK, &api.Endpoints{}})
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
		serverResponse{http.StatusInternalServerError, &api.ServiceList{}},
		serverResponse{http.StatusOK, &api.Endpoints{}})
	defer testServer.Close()
	client := client.NewOrDie(&client.Config{Host: testServer.URL, Version: testapi.Version()})
	endpoints := NewEndpointController(client)
	if err := endpoints.SyncServiceEndpoints(); err == nil {
		t.Errorf("unexpected non-error")
	}
}

func TestSyncEndpointsItemsPreserveNoSelector(t *testing.T) {
	serviceList := api.ServiceList{
		Items: []api.Service{
			{
				ObjectMeta: api.ObjectMeta{Name: "foo"},
				Spec:       api.ServiceSpec{},
			},
		},
	}
	testServer, endpointsHandler := makeTestServer(t,
		serverResponse{http.StatusOK, newPodList(0)},
		serverResponse{http.StatusOK, &serviceList},
		serverResponse{http.StatusOK, &api.Endpoints{
			ObjectMeta: api.ObjectMeta{
				Name:            "foo",
				ResourceVersion: "1",
			},
			Protocol:  api.ProtocolTCP,
			Endpoints: []api.Endpoint{{IP: "6.7.8.9", Port: 1000}},
		}})
	defer testServer.Close()
	client := client.NewOrDie(&client.Config{Host: testServer.URL, Version: testapi.Version()})
	endpoints := NewEndpointController(client)
	if err := endpoints.SyncServiceEndpoints(); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	endpointsHandler.ValidateRequestCount(t, 0)
}

func TestSyncEndpointsProtocolTCP(t *testing.T) {
	serviceList := api.ServiceList{
		Items: []api.Service{
			{
				ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: "other"},
				Spec: api.ServiceSpec{
					Selector: map[string]string{},
					Protocol: api.ProtocolTCP,
				},
			},
		},
	}
	testServer, endpointsHandler := makeTestServer(t,
		serverResponse{http.StatusOK, newPodList(0)},
		serverResponse{http.StatusOK, &serviceList},
		serverResponse{http.StatusOK, &api.Endpoints{
			ObjectMeta: api.ObjectMeta{
				Name:            "foo",
				ResourceVersion: "1",
			},
			Protocol:  api.ProtocolTCP,
			Endpoints: []api.Endpoint{{IP: "6.7.8.9", Port: 1000}},
		}})
	defer testServer.Close()
	client := client.NewOrDie(&client.Config{Host: testServer.URL, Version: testapi.Version()})
	endpoints := NewEndpointController(client)
	if err := endpoints.SyncServiceEndpoints(); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	endpointsHandler.ValidateRequestCount(t, 0)
}

func TestSyncEndpointsProtocolUDP(t *testing.T) {
	serviceList := api.ServiceList{
		Items: []api.Service{
			{
				ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: "other"},
				Spec: api.ServiceSpec{
					Selector: map[string]string{},
					Protocol: api.ProtocolUDP,
				},
			},
		},
	}
	testServer, endpointsHandler := makeTestServer(t,
		serverResponse{http.StatusOK, newPodList(0)},
		serverResponse{http.StatusOK, &serviceList},
		serverResponse{http.StatusOK, &api.Endpoints{
			ObjectMeta: api.ObjectMeta{
				Name:            "foo",
				ResourceVersion: "1",
			},
			Protocol:  api.ProtocolUDP,
			Endpoints: []api.Endpoint{{IP: "6.7.8.9", Port: 1000}},
		}})
	defer testServer.Close()
	client := client.NewOrDie(&client.Config{Host: testServer.URL, Version: testapi.Version()})
	endpoints := NewEndpointController(client)
	if err := endpoints.SyncServiceEndpoints(); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	endpointsHandler.ValidateRequestCount(t, 0)
}

func TestSyncEndpointsItemsEmptySelectorSelectsAll(t *testing.T) {
	serviceList := api.ServiceList{
		Items: []api.Service{
			{
				ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: "other"},
				Spec: api.ServiceSpec{
					Selector: map[string]string{},
				},
			},
		},
	}
	testServer, endpointsHandler := makeTestServer(t,
		serverResponse{http.StatusOK, newPodList(1)},
		serverResponse{http.StatusOK, &serviceList},
		serverResponse{http.StatusOK, &api.Endpoints{
			ObjectMeta: api.ObjectMeta{
				Name:            "foo",
				ResourceVersion: "1",
			},
			Protocol:  api.ProtocolTCP,
			Endpoints: []api.Endpoint{},
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
		Protocol: api.ProtocolTCP,
		Endpoints: []api.Endpoint{{
			IP:   "1.2.3.4",
			Port: 8080,
			TargetRef: &api.ObjectReference{
				Kind: "Pod",
				Name: "pod0",
			},
		}},
	})
	endpointsHandler.ValidateRequest(t, "/api/"+testapi.Version()+"/endpoints/foo?namespace=other", "PUT", &data)
}

func TestSyncEndpointsItemsPreexisting(t *testing.T) {
	serviceList := api.ServiceList{
		Items: []api.Service{
			{
				ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: "bar"},
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
		serverResponse{http.StatusOK, &serviceList},
		serverResponse{http.StatusOK, &api.Endpoints{
			ObjectMeta: api.ObjectMeta{
				Name:            "foo",
				ResourceVersion: "1",
			},
			Protocol:  api.ProtocolTCP,
			Endpoints: []api.Endpoint{{IP: "6.7.8.9", Port: 1000}},
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
		Protocol: api.ProtocolTCP,
		Endpoints: []api.Endpoint{{
			IP:   "1.2.3.4",
			Port: 8080,
			TargetRef: &api.ObjectReference{
				Kind: "Pod",
				Name: "pod0",
			},
		}},
	})
	endpointsHandler.ValidateRequest(t, "/api/"+testapi.Version()+"/endpoints/foo?namespace=bar", "PUT", &data)
}

func TestSyncEndpointsItemsPreexistingIdentical(t *testing.T) {
	serviceList := api.ServiceList{
		Items: []api.Service{
			{
				ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: api.NamespaceDefault},
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
		serverResponse{http.StatusOK, &serviceList},
		serverResponse{http.StatusOK, &api.Endpoints{
			ObjectMeta: api.ObjectMeta{
				ResourceVersion: "1",
			},
			Protocol: api.ProtocolTCP,
			Endpoints: []api.Endpoint{{
				IP:   "1.2.3.4",
				Port: 8080,
				TargetRef: &api.ObjectReference{
					Kind: "Pod",
					Name: "pod0",
				},
			}},
		}})
	defer testServer.Close()
	client := client.NewOrDie(&client.Config{Host: testServer.URL, Version: testapi.Version()})
	endpoints := NewEndpointController(client)
	if err := endpoints.SyncServiceEndpoints(); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	endpointsHandler.ValidateRequest(t, "/api/"+testapi.Version()+"/endpoints/foo?namespace=default", "GET", nil)
}

func TestSyncEndpointsItems(t *testing.T) {
	serviceList := api.ServiceList{
		Items: []api.Service{
			{
				ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: "other"},
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
		serverResponse{http.StatusOK, &serviceList},
		serverResponse{http.StatusOK, &api.Endpoints{}})
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
		Protocol: api.ProtocolTCP,
		Endpoints: []api.Endpoint{{
			IP:   "1.2.3.4",
			Port: 8080,
			TargetRef: &api.ObjectReference{
				Kind: "Pod",
				Name: "pod0",
			},
		}},
	})
	endpointsHandler.ValidateRequest(t, "/api/"+testapi.Version()+"/endpoints?namespace=other", "POST", &data)
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
		serverResponse{http.StatusInternalServerError, &api.PodList{}},
		serverResponse{http.StatusOK, &serviceList},
		serverResponse{http.StatusOK, &api.Endpoints{}})
	defer testServer.Close()
	client := client.NewOrDie(&client.Config{Host: testServer.URL, Version: testapi.Version()})
	endpoints := NewEndpointController(client)
	if err := endpoints.SyncServiceEndpoints(); err == nil {
		t.Error("Unexpected non-error")
	}
}
