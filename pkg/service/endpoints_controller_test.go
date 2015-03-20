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
	endptspkg "github.com/GoogleCloudPlatform/kubernetes/pkg/api/endpoints"
	_ "github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

func newPodList(nPods int, nPorts int) *api.PodList {
	pods := []api.Pod{}
	for i := 0; i < nPods; i++ {
		p := api.Pod{
			TypeMeta:   api.TypeMeta{APIVersion: testapi.Version()},
			ObjectMeta: api.ObjectMeta{Name: fmt.Sprintf("pod%d", i)},
			Spec: api.PodSpec{
				Containers: []api.Container{{Ports: []api.ContainerPort{}}},
			},
			Status: api.PodStatus{
				PodIP: fmt.Sprintf("1.2.3.%d", 4+i),
				Conditions: []api.PodCondition{
					{
						Type:   api.PodReady,
						Status: api.ConditionTrue,
					},
				},
			},
		}
		for j := 0; j < nPorts; j++ {
			p.Spec.Containers[0].Ports = append(p.Spec.Containers[0].Ports, api.ContainerPort{ContainerPort: 8080 + j})
		}
		pods = append(pods, p)
	}
	return &api.PodList{
		TypeMeta: api.TypeMeta{APIVersion: testapi.Version(), Kind: "PodList"},
		Items:    pods,
	}
}

func TestFindPort(t *testing.T) {
	servicePort := 999
	testCases := []struct {
		name       string
		containers []api.Container
		port       util.IntOrString
		expected   int
		pass       bool
	}{{
		name:       "valid int, no ports",
		containers: []api.Container{{}},
		port:       util.NewIntOrStringFromInt(93),
		expected:   93,
		pass:       true,
	}, {
		name: "valid int, with ports",
		containers: []api.Container{{Ports: []api.ContainerPort{{
			Name:          "",
			ContainerPort: 11,
			Protocol:      "TCP",
		}, {
			Name:          "p",
			ContainerPort: 22,
			Protocol:      "TCP",
		}}}},
		port:     util.NewIntOrStringFromInt(93),
		expected: 93,
		pass:     true,
	}, {
		name:       "zero int, no ports",
		containers: []api.Container{{}},
		port:       util.NewIntOrStringFromInt(0),
		expected:   servicePort,
		pass:       true,
	}, {
		name: "zero int, one ctr with ports",
		containers: []api.Container{{Ports: []api.ContainerPort{{
			Name:          "",
			ContainerPort: 11,
			Protocol:      "UDP",
		}, {
			Name:          "p",
			ContainerPort: 22,
			Protocol:      "TCP",
		}}}},
		port:     util.NewIntOrStringFromInt(0),
		expected: 22,
		pass:     true,
	}, {
		name: "zero int, two ctr with ports",
		containers: []api.Container{{}, {Ports: []api.ContainerPort{{
			Name:          "",
			ContainerPort: 11,
			Protocol:      "UDP",
		}, {
			Name:          "p",
			ContainerPort: 22,
			Protocol:      "TCP",
		}}}},
		port:     util.NewIntOrStringFromInt(0),
		expected: 22,
		pass:     true,
	}, {
		name:       "empty str, no ports",
		containers: []api.Container{{}},
		port:       util.NewIntOrStringFromString(""),
		expected:   servicePort,
		pass:       true,
	}, {
		name: "empty str, one ctr with ports",
		containers: []api.Container{{Ports: []api.ContainerPort{{
			Name:          "",
			ContainerPort: 11,
			Protocol:      "UDP",
		}, {
			Name:          "p",
			ContainerPort: 22,
			Protocol:      "TCP",
		}}}},
		port:     util.NewIntOrStringFromString(""),
		expected: 22,
		pass:     true,
	}, {
		name: "empty str, two ctr with ports",
		containers: []api.Container{{}, {Ports: []api.ContainerPort{{
			Name:          "",
			ContainerPort: 11,
			Protocol:      "UDP",
		}, {
			Name:          "p",
			ContainerPort: 22,
			Protocol:      "TCP",
		}}}},
		port:     util.NewIntOrStringFromString(""),
		expected: 22,
		pass:     true,
	}, {
		name:       "valid str, no ports",
		containers: []api.Container{{}},
		port:       util.NewIntOrStringFromString("p"),
		expected:   0,
		pass:       false,
	}, {
		name: "valid str, one ctr with ports",
		containers: []api.Container{{Ports: []api.ContainerPort{{
			Name:          "",
			ContainerPort: 11,
			Protocol:      "UDP",
		}, {
			Name:          "p",
			ContainerPort: 22,
			Protocol:      "TCP",
		}, {
			Name:          "q",
			ContainerPort: 33,
			Protocol:      "TCP",
		}}}},
		port:     util.NewIntOrStringFromString("q"),
		expected: 33,
		pass:     true,
	}, {
		name: "valid str, two ctr with ports",
		containers: []api.Container{{}, {Ports: []api.ContainerPort{{
			Name:          "",
			ContainerPort: 11,
			Protocol:      "UDP",
		}, {
			Name:          "p",
			ContainerPort: 22,
			Protocol:      "TCP",
		}, {
			Name:          "q",
			ContainerPort: 33,
			Protocol:      "TCP",
		}}}},
		port:     util.NewIntOrStringFromString("q"),
		expected: 33,
		pass:     true,
	}}

	for _, tc := range testCases {
		port, err := findPort(&api.Pod{Spec: api.PodSpec{Containers: tc.containers}},
			&api.Service{Spec: api.ServiceSpec{Protocol: "TCP", Port: servicePort, TargetPort: tc.port}})
		if err != nil && tc.pass {
			t.Errorf("unexpected error for %s: %v", tc.name, err)
		}
		if err == nil && !tc.pass {
			t.Errorf("unexpected non-error for %s: %d", tc.name, port)
		}
		if port != tc.expected {
			t.Errorf("wrong result for %s: expected %d, got %d", tc.name, tc.expected, port)
		}
	}
}

type serverResponse struct {
	statusCode int
	obj        interface{}
}

func makeTestServer(t *testing.T, namespace string, podResponse, serviceResponse, endpointsResponse serverResponse) (*httptest.Server, *util.FakeHandler) {
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
	mux.Handle(testapi.ResourcePath("pods", namespace, ""), &fakePodHandler)
	mux.Handle(testapi.ResourcePath("services", "", ""), &fakeServiceHandler)
	mux.Handle(testapi.ResourcePath("endpoints", namespace, ""), &fakeEndpointsHandler)
	mux.Handle(testapi.ResourcePath("endpoints/", namespace, ""), &fakeEndpointsHandler)
	mux.HandleFunc("/", func(res http.ResponseWriter, req *http.Request) {
		t.Errorf("unexpected request: %v", req.RequestURI)
		res.WriteHeader(http.StatusNotFound)
	})
	return httptest.NewServer(mux), &fakeEndpointsHandler
}

func TestSyncEndpointsEmpty(t *testing.T) {
	testServer, _ := makeTestServer(t, api.NamespaceDefault,
		serverResponse{http.StatusOK, newPodList(0, 0)},
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
	testServer, _ := makeTestServer(t, api.NamespaceDefault,
		serverResponse{http.StatusOK, newPodList(0, 0)},
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
	testServer, endpointsHandler := makeTestServer(t, api.NamespaceDefault,
		serverResponse{http.StatusOK, newPodList(0, 0)},
		serverResponse{http.StatusOK, &serviceList},
		serverResponse{http.StatusOK, &api.Endpoints{
			ObjectMeta: api.ObjectMeta{
				Name:            "foo",
				ResourceVersion: "1",
			},
			Subsets: []api.EndpointSubset{{
				Addresses: []api.EndpointAddress{{IP: "6.7.8.9"}},
				Ports:     []api.EndpointPort{{Port: 1000}},
			}},
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
	testServer, endpointsHandler := makeTestServer(t, "other",
		serverResponse{http.StatusOK, newPodList(0, 0)},
		serverResponse{http.StatusOK, &serviceList},
		serverResponse{http.StatusOK, &api.Endpoints{
			ObjectMeta: api.ObjectMeta{
				Name:            "foo",
				ResourceVersion: "1",
			},
			Subsets: []api.EndpointSubset{{
				Addresses: []api.EndpointAddress{{IP: "6.7.8.9"}},
				Ports:     []api.EndpointPort{{Port: 1000, Protocol: "TCP"}},
			}},
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
	testServer, endpointsHandler := makeTestServer(t, "other",
		serverResponse{http.StatusOK, newPodList(0, 0)},
		serverResponse{http.StatusOK, &serviceList},
		serverResponse{http.StatusOK, &api.Endpoints{
			ObjectMeta: api.ObjectMeta{
				Name:            "foo",
				ResourceVersion: "1",
			},
			Subsets: []api.EndpointSubset{{
				Addresses: []api.EndpointAddress{{IP: "6.7.8.9"}},
				Ports:     []api.EndpointPort{{Port: 1000, Protocol: "UDP"}},
			}},
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
	testServer, endpointsHandler := makeTestServer(t, "other",
		serverResponse{http.StatusOK, newPodList(1, 1)},
		serverResponse{http.StatusOK, &serviceList},
		serverResponse{http.StatusOK, &api.Endpoints{
			ObjectMeta: api.ObjectMeta{
				Name:            "foo",
				ResourceVersion: "1",
			},
			Subsets: []api.EndpointSubset{},
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
		Subsets: []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{IP: "1.2.3.4", TargetRef: &api.ObjectReference{Kind: "Pod", Name: "pod0"}}},
			Ports:     []api.EndpointPort{{Port: 8080, Protocol: "TCP"}},
		}},
	})
	endpointsHandler.ValidateRequest(t, testapi.ResourcePathWithQueryParams("endpoints", "other", "foo"), "PUT", &data)
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
	testServer, endpointsHandler := makeTestServer(t, "bar",
		serverResponse{http.StatusOK, newPodList(1, 1)},
		serverResponse{http.StatusOK, &serviceList},
		serverResponse{http.StatusOK, &api.Endpoints{
			ObjectMeta: api.ObjectMeta{
				Name:            "foo",
				ResourceVersion: "1",
			},
			Subsets: []api.EndpointSubset{{
				Addresses: []api.EndpointAddress{{IP: "6.7.8.9"}},
				Ports:     []api.EndpointPort{{Port: 1000}},
			}},
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
		Subsets: []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{IP: "1.2.3.4", TargetRef: &api.ObjectReference{Kind: "Pod", Name: "pod0"}}},
			Ports:     []api.EndpointPort{{Port: 8080, Protocol: "TCP"}},
		}},
	})
	endpointsHandler.ValidateRequest(t, testapi.ResourcePathWithQueryParams("endpoints", "bar", "foo"), "PUT", &data)
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
	testServer, endpointsHandler := makeTestServer(t, api.NamespaceDefault,
		serverResponse{http.StatusOK, newPodList(1, 1)},
		serverResponse{http.StatusOK, &serviceList},
		serverResponse{http.StatusOK, &api.Endpoints{
			ObjectMeta: api.ObjectMeta{
				ResourceVersion: "1",
			},
			Subsets: []api.EndpointSubset{{
				Addresses: []api.EndpointAddress{{IP: "1.2.3.4", TargetRef: &api.ObjectReference{Kind: "Pod", Name: "pod0"}}},
				Ports:     []api.EndpointPort{{Port: 8080, Protocol: "TCP"}},
			}},
		}})
	defer testServer.Close()
	client := client.NewOrDie(&client.Config{Host: testServer.URL, Version: testapi.Version()})
	endpoints := NewEndpointController(client)
	if err := endpoints.SyncServiceEndpoints(); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	endpointsHandler.ValidateRequest(t, testapi.ResourcePathWithQueryParams("endpoints", api.NamespaceDefault, "foo"), "GET", nil)
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
	testServer, endpointsHandler := makeTestServer(t, "other",
		serverResponse{http.StatusOK, newPodList(3, 2)},
		serverResponse{http.StatusOK, &serviceList},
		serverResponse{http.StatusOK, &api.Endpoints{}})
	defer testServer.Close()
	client := client.NewOrDie(&client.Config{Host: testServer.URL, Version: testapi.Version()})
	endpoints := NewEndpointController(client)
	if err := endpoints.SyncServiceEndpoints(); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	expectedSubsets := []api.EndpointSubset{{
		Addresses: []api.EndpointAddress{
			{IP: "1.2.3.4", TargetRef: &api.ObjectReference{Kind: "Pod", Name: "pod0"}},
			{IP: "1.2.3.5", TargetRef: &api.ObjectReference{Kind: "Pod", Name: "pod1"}},
			{IP: "1.2.3.6", TargetRef: &api.ObjectReference{Kind: "Pod", Name: "pod2"}},
		},
		Ports: []api.EndpointPort{
			{Port: 8080, Protocol: "TCP"},
		},
	}}
	data := runtime.EncodeOrDie(testapi.Codec(), &api.Endpoints{
		ObjectMeta: api.ObjectMeta{
			ResourceVersion: "",
		},
		Subsets: endptspkg.SortSubsets(expectedSubsets),
	})
	// endpointsHandler should get 2 requests - one for "GET" and the next for "POST".
	endpointsHandler.ValidateRequestCount(t, 2)
	endpointsHandler.ValidateRequest(t, testapi.ResourcePathWithQueryParams("endpoints", "other", ""), "POST", &data)
}

func TestSyncEndpointsPodError(t *testing.T) {
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
	testServer, _ := makeTestServer(t, api.NamespaceDefault,
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
