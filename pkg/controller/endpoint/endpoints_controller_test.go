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

package endpointcontroller

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	endptspkg "k8s.io/kubernetes/pkg/api/endpoints"
	_ "k8s.io/kubernetes/pkg/api/latest"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/cache"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
)

func addPods(store cache.Store, namespace string, nPods int, nPorts int, nNotReady int) {
	for i := 0; i < nPods+nNotReady; i++ {
		p := &api.Pod{
			TypeMeta: unversioned.TypeMeta{APIVersion: testapi.Default.Version()},
			ObjectMeta: api.ObjectMeta{
				Namespace: namespace,
				Name:      fmt.Sprintf("pod%d", i),
				Labels:    map[string]string{"foo": "bar"},
			},
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
		if i >= nPods {
			p.Status.Conditions[0].Status = api.ConditionFalse
		}
		for j := 0; j < nPorts; j++ {
			p.Spec.Containers[0].Ports = append(p.Spec.Containers[0].Ports,
				api.ContainerPort{Name: fmt.Sprintf("port%d", i), ContainerPort: 8080 + j})
		}
		store.Add(p)
	}
}

func TestFindPort(t *testing.T) {
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
			&api.ServicePort{Protocol: "TCP", TargetPort: tc.port})
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

func makeTestServer(t *testing.T, namespace string, endpointsResponse serverResponse) (*httptest.Server, *util.FakeHandler) {
	fakeEndpointsHandler := util.FakeHandler{
		StatusCode:   endpointsResponse.statusCode,
		ResponseBody: runtime.EncodeOrDie(testapi.Default.Codec(), endpointsResponse.obj.(runtime.Object)),
	}
	mux := http.NewServeMux()
	mux.Handle(testapi.Default.ResourcePath("endpoints", namespace, ""), &fakeEndpointsHandler)
	mux.Handle(testapi.Default.ResourcePath("endpoints/", namespace, ""), &fakeEndpointsHandler)
	mux.HandleFunc("/", func(res http.ResponseWriter, req *http.Request) {
		t.Errorf("unexpected request: %v", req.RequestURI)
		res.WriteHeader(http.StatusNotFound)
	})
	return httptest.NewServer(mux), &fakeEndpointsHandler
}

func TestSyncEndpointsItemsPreserveNoSelector(t *testing.T) {
	ns := api.NamespaceDefault
	testServer, endpointsHandler := makeTestServer(t, ns,
		serverResponse{http.StatusOK, &api.Endpoints{
			ObjectMeta: api.ObjectMeta{
				Name:            "foo",
				Namespace:       ns,
				ResourceVersion: "1",
			},
			Subsets: []api.EndpointSubset{{
				Addresses: []api.EndpointAddress{{IP: "6.7.8.9"}},
				Ports:     []api.EndpointPort{{Port: 1000}},
			}},
		}})
	defer testServer.Close()
	client := client.NewOrDie(&client.Config{Host: testServer.URL, Version: testapi.Default.Version()})
	endpoints := NewEndpointController(client, controller.NoResyncPeriodFunc)
	endpoints.serviceStore.Store.Add(&api.Service{
		ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: ns},
		Spec:       api.ServiceSpec{Ports: []api.ServicePort{{Port: 80}}},
	})
	endpoints.syncService(ns + "/foo")
	endpointsHandler.ValidateRequestCount(t, 0)
}

func TestCheckLeftoverEndpoints(t *testing.T) {
	ns := api.NamespaceDefault
	// Note that this requests *all* endpoints, therefore the NamespaceAll
	// below.
	testServer, _ := makeTestServer(t, api.NamespaceAll,
		serverResponse{http.StatusOK, &api.EndpointsList{
			ListMeta: unversioned.ListMeta{
				ResourceVersion: "1",
			},
			Items: []api.Endpoints{{
				ObjectMeta: api.ObjectMeta{
					Name:            "foo",
					Namespace:       ns,
					ResourceVersion: "1",
				},
				Subsets: []api.EndpointSubset{{
					Addresses: []api.EndpointAddress{{IP: "6.7.8.9"}},
					Ports:     []api.EndpointPort{{Port: 1000}},
				}},
			}},
		}})
	defer testServer.Close()
	client := client.NewOrDie(&client.Config{Host: testServer.URL, Version: testapi.Default.Version()})
	endpoints := NewEndpointController(client, controller.NoResyncPeriodFunc)
	endpoints.checkLeftoverEndpoints()

	if e, a := 1, endpoints.queue.Len(); e != a {
		t.Fatalf("Expected %v, got %v", e, a)
	}
	got, _ := endpoints.queue.Get()
	if e, a := ns+"/foo", got; e != a {
		t.Errorf("Expected %v, got %v", e, a)
	}
}

func TestSyncEndpointsProtocolTCP(t *testing.T) {
	ns := "other"
	testServer, endpointsHandler := makeTestServer(t, ns,
		serverResponse{http.StatusOK, &api.Endpoints{
			ObjectMeta: api.ObjectMeta{
				Name:            "foo",
				Namespace:       ns,
				ResourceVersion: "1",
			},
			Subsets: []api.EndpointSubset{{
				Addresses: []api.EndpointAddress{{IP: "6.7.8.9"}},
				Ports:     []api.EndpointPort{{Port: 1000, Protocol: "TCP"}},
			}},
		}})
	defer testServer.Close()
	client := client.NewOrDie(&client.Config{Host: testServer.URL, Version: testapi.Default.Version()})
	endpoints := NewEndpointController(client, controller.NoResyncPeriodFunc)
	endpoints.serviceStore.Store.Add(&api.Service{
		ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: ns},
		Spec: api.ServiceSpec{
			Selector: map[string]string{},
			Ports:    []api.ServicePort{{Port: 80}},
		},
	})
	endpoints.syncService(ns + "/foo")
	endpointsHandler.ValidateRequestCount(t, 0)
}

func TestSyncEndpointsProtocolUDP(t *testing.T) {
	ns := "other"
	testServer, endpointsHandler := makeTestServer(t, ns,
		serverResponse{http.StatusOK, &api.Endpoints{
			ObjectMeta: api.ObjectMeta{
				Name:            "foo",
				Namespace:       ns,
				ResourceVersion: "1",
			},
			Subsets: []api.EndpointSubset{{
				Addresses: []api.EndpointAddress{{IP: "6.7.8.9"}},
				Ports:     []api.EndpointPort{{Port: 1000, Protocol: "UDP"}},
			}},
		}})
	defer testServer.Close()
	client := client.NewOrDie(&client.Config{Host: testServer.URL, Version: testapi.Default.Version()})
	endpoints := NewEndpointController(client, controller.NoResyncPeriodFunc)
	endpoints.serviceStore.Store.Add(&api.Service{
		ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: ns},
		Spec: api.ServiceSpec{
			Selector: map[string]string{},
			Ports:    []api.ServicePort{{Port: 80}},
		},
	})
	endpoints.syncService(ns + "/foo")
	endpointsHandler.ValidateRequestCount(t, 0)
}

func TestSyncEndpointsItemsEmptySelectorSelectsAll(t *testing.T) {
	ns := "other"
	testServer, endpointsHandler := makeTestServer(t, ns,
		serverResponse{http.StatusOK, &api.Endpoints{
			ObjectMeta: api.ObjectMeta{
				Name:            "foo",
				Namespace:       ns,
				ResourceVersion: "1",
			},
			Subsets: []api.EndpointSubset{},
		}})
	defer testServer.Close()
	client := client.NewOrDie(&client.Config{Host: testServer.URL, Version: testapi.Default.Version()})
	endpoints := NewEndpointController(client, controller.NoResyncPeriodFunc)
	addPods(endpoints.podStore.Store, ns, 1, 1, 0)
	endpoints.serviceStore.Store.Add(&api.Service{
		ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: ns},
		Spec: api.ServiceSpec{
			Selector: map[string]string{},
			Ports:    []api.ServicePort{{Port: 80, Protocol: "TCP", TargetPort: util.NewIntOrStringFromInt(8080)}},
		},
	})
	endpoints.syncService(ns + "/foo")
	data := runtime.EncodeOrDie(testapi.Default.Codec(), &api.Endpoints{
		ObjectMeta: api.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
		},
		Subsets: []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{IP: "1.2.3.4", TargetRef: &api.ObjectReference{Kind: "Pod", Name: "pod0", Namespace: ns}}},
			Ports:     []api.EndpointPort{{Port: 8080, Protocol: "TCP"}},
		}},
	})
	endpointsHandler.ValidateRequest(t, testapi.Default.ResourcePath("endpoints", ns, "foo"), "PUT", &data)
}

func TestSyncEndpointsItemsEmptySelectorSelectsAllNotReady(t *testing.T) {
	ns := "other"
	testServer, endpointsHandler := makeTestServer(t, ns,
		serverResponse{http.StatusOK, &api.Endpoints{
			ObjectMeta: api.ObjectMeta{
				Name:            "foo",
				Namespace:       ns,
				ResourceVersion: "1",
			},
			Subsets: []api.EndpointSubset{},
		}})
	defer testServer.Close()
	client := client.NewOrDie(&client.Config{Host: testServer.URL, Version: testapi.Default.Version()})
	endpoints := NewEndpointController(client, controller.NoResyncPeriodFunc)
	addPods(endpoints.podStore.Store, ns, 0, 1, 1)
	endpoints.serviceStore.Store.Add(&api.Service{
		ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: ns},
		Spec: api.ServiceSpec{
			Selector: map[string]string{},
			Ports:    []api.ServicePort{{Port: 80, Protocol: "TCP", TargetPort: util.NewIntOrStringFromInt(8080)}},
		},
	})
	endpoints.syncService(ns + "/foo")
	data := runtime.EncodeOrDie(testapi.Default.Codec(), &api.Endpoints{
		ObjectMeta: api.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
		},
		Subsets: []api.EndpointSubset{{
			NotReadyAddresses: []api.EndpointAddress{{IP: "1.2.3.4", TargetRef: &api.ObjectReference{Kind: "Pod", Name: "pod0", Namespace: ns}}},
			Ports:             []api.EndpointPort{{Port: 8080, Protocol: "TCP"}},
		}},
	})
	endpointsHandler.ValidateRequest(t, testapi.Default.ResourcePath("endpoints", ns, "foo"), "PUT", &data)
}

func TestSyncEndpointsItemsEmptySelectorSelectsAllMixed(t *testing.T) {
	ns := "other"
	testServer, endpointsHandler := makeTestServer(t, ns,
		serverResponse{http.StatusOK, &api.Endpoints{
			ObjectMeta: api.ObjectMeta{
				Name:            "foo",
				Namespace:       ns,
				ResourceVersion: "1",
			},
			Subsets: []api.EndpointSubset{},
		}})
	defer testServer.Close()
	client := client.NewOrDie(&client.Config{Host: testServer.URL, Version: testapi.Default.Version()})
	endpoints := NewEndpointController(client, controller.NoResyncPeriodFunc)
	addPods(endpoints.podStore.Store, ns, 1, 1, 1)
	endpoints.serviceStore.Store.Add(&api.Service{
		ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: ns},
		Spec: api.ServiceSpec{
			Selector: map[string]string{},
			Ports:    []api.ServicePort{{Port: 80, Protocol: "TCP", TargetPort: util.NewIntOrStringFromInt(8080)}},
		},
	})
	endpoints.syncService(ns + "/foo")
	data := runtime.EncodeOrDie(testapi.Default.Codec(), &api.Endpoints{
		ObjectMeta: api.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
		},
		Subsets: []api.EndpointSubset{{
			Addresses:         []api.EndpointAddress{{IP: "1.2.3.4", TargetRef: &api.ObjectReference{Kind: "Pod", Name: "pod0", Namespace: ns}}},
			NotReadyAddresses: []api.EndpointAddress{{IP: "1.2.3.5", TargetRef: &api.ObjectReference{Kind: "Pod", Name: "pod1", Namespace: ns}}},
			Ports:             []api.EndpointPort{{Port: 8080, Protocol: "TCP"}},
		}},
	})
	endpointsHandler.ValidateRequest(t, testapi.Default.ResourcePath("endpoints", ns, "foo"), "PUT", &data)
}

func TestSyncEndpointsItemsPreexisting(t *testing.T) {
	ns := "bar"
	testServer, endpointsHandler := makeTestServer(t, ns,
		serverResponse{http.StatusOK, &api.Endpoints{
			ObjectMeta: api.ObjectMeta{
				Name:            "foo",
				Namespace:       ns,
				ResourceVersion: "1",
			},
			Subsets: []api.EndpointSubset{{
				Addresses: []api.EndpointAddress{{IP: "6.7.8.9"}},
				Ports:     []api.EndpointPort{{Port: 1000}},
			}},
		}})
	defer testServer.Close()
	client := client.NewOrDie(&client.Config{Host: testServer.URL, Version: testapi.Default.Version()})
	endpoints := NewEndpointController(client, controller.NoResyncPeriodFunc)
	addPods(endpoints.podStore.Store, ns, 1, 1, 0)
	endpoints.serviceStore.Store.Add(&api.Service{
		ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: ns},
		Spec: api.ServiceSpec{
			Selector: map[string]string{"foo": "bar"},
			Ports:    []api.ServicePort{{Port: 80, Protocol: "TCP", TargetPort: util.NewIntOrStringFromInt(8080)}},
		},
	})
	endpoints.syncService(ns + "/foo")
	data := runtime.EncodeOrDie(testapi.Default.Codec(), &api.Endpoints{
		ObjectMeta: api.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
		},
		Subsets: []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{IP: "1.2.3.4", TargetRef: &api.ObjectReference{Kind: "Pod", Name: "pod0", Namespace: ns}}},
			Ports:     []api.EndpointPort{{Port: 8080, Protocol: "TCP"}},
		}},
	})
	endpointsHandler.ValidateRequest(t, testapi.Default.ResourcePath("endpoints", ns, "foo"), "PUT", &data)
}

func TestSyncEndpointsItemsPreexistingIdentical(t *testing.T) {
	ns := api.NamespaceDefault
	testServer, endpointsHandler := makeTestServer(t, api.NamespaceDefault,
		serverResponse{http.StatusOK, &api.Endpoints{
			ObjectMeta: api.ObjectMeta{
				ResourceVersion: "1",
				Name:            "foo",
				Namespace:       ns,
			},
			Subsets: []api.EndpointSubset{{
				Addresses: []api.EndpointAddress{{IP: "1.2.3.4", TargetRef: &api.ObjectReference{Kind: "Pod", Name: "pod0", Namespace: ns}}},
				Ports:     []api.EndpointPort{{Port: 8080, Protocol: "TCP"}},
			}},
		}})
	defer testServer.Close()
	client := client.NewOrDie(&client.Config{Host: testServer.URL, Version: testapi.Default.Version()})
	endpoints := NewEndpointController(client, controller.NoResyncPeriodFunc)
	addPods(endpoints.podStore.Store, api.NamespaceDefault, 1, 1, 0)
	endpoints.serviceStore.Store.Add(&api.Service{
		ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: api.NamespaceDefault},
		Spec: api.ServiceSpec{
			Selector: map[string]string{"foo": "bar"},
			Ports:    []api.ServicePort{{Port: 80, Protocol: "TCP", TargetPort: util.NewIntOrStringFromInt(8080)}},
		},
	})
	endpoints.syncService(ns + "/foo")
	endpointsHandler.ValidateRequest(t, testapi.Default.ResourcePath("endpoints", api.NamespaceDefault, "foo"), "GET", nil)
}

func TestSyncEndpointsItems(t *testing.T) {
	ns := "other"
	testServer, endpointsHandler := makeTestServer(t, ns,
		serverResponse{http.StatusOK, &api.Endpoints{}})
	defer testServer.Close()
	client := client.NewOrDie(&client.Config{Host: testServer.URL, Version: testapi.Default.Version()})
	endpoints := NewEndpointController(client, controller.NoResyncPeriodFunc)
	addPods(endpoints.podStore.Store, ns, 3, 2, 0)
	addPods(endpoints.podStore.Store, "blah", 5, 2, 0) // make sure these aren't found!
	endpoints.serviceStore.Store.Add(&api.Service{
		ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: ns},
		Spec: api.ServiceSpec{
			Selector: map[string]string{"foo": "bar"},
			Ports: []api.ServicePort{
				{Name: "port0", Port: 80, Protocol: "TCP", TargetPort: util.NewIntOrStringFromInt(8080)},
				{Name: "port1", Port: 88, Protocol: "TCP", TargetPort: util.NewIntOrStringFromInt(8088)},
			},
		},
	})
	endpoints.syncService("other/foo")
	expectedSubsets := []api.EndpointSubset{{
		Addresses: []api.EndpointAddress{
			{IP: "1.2.3.4", TargetRef: &api.ObjectReference{Kind: "Pod", Name: "pod0", Namespace: ns}},
			{IP: "1.2.3.5", TargetRef: &api.ObjectReference{Kind: "Pod", Name: "pod1", Namespace: ns}},
			{IP: "1.2.3.6", TargetRef: &api.ObjectReference{Kind: "Pod", Name: "pod2", Namespace: ns}},
		},
		Ports: []api.EndpointPort{
			{Name: "port0", Port: 8080, Protocol: "TCP"},
			{Name: "port1", Port: 8088, Protocol: "TCP"},
		},
	}}
	data := runtime.EncodeOrDie(testapi.Default.Codec(), &api.Endpoints{
		ObjectMeta: api.ObjectMeta{
			ResourceVersion: "",
		},
		Subsets: endptspkg.SortSubsets(expectedSubsets),
	})
	// endpointsHandler should get 2 requests - one for "GET" and the next for "POST".
	endpointsHandler.ValidateRequestCount(t, 2)
	endpointsHandler.ValidateRequest(t, testapi.Default.ResourcePath("endpoints", ns, ""), "POST", &data)
}

func TestSyncEndpointsItemsWithLabels(t *testing.T) {
	ns := "other"
	testServer, endpointsHandler := makeTestServer(t, ns,
		serverResponse{http.StatusOK, &api.Endpoints{}})
	defer testServer.Close()
	client := client.NewOrDie(&client.Config{Host: testServer.URL, Version: testapi.Default.Version()})
	endpoints := NewEndpointController(client, controller.NoResyncPeriodFunc)
	addPods(endpoints.podStore.Store, ns, 3, 2, 0)
	serviceLabels := map[string]string{"foo": "bar"}
	endpoints.serviceStore.Store.Add(&api.Service{
		ObjectMeta: api.ObjectMeta{
			Name:      "foo",
			Namespace: ns,
			Labels:    serviceLabels,
		},
		Spec: api.ServiceSpec{
			Selector: map[string]string{"foo": "bar"},
			Ports: []api.ServicePort{
				{Name: "port0", Port: 80, Protocol: "TCP", TargetPort: util.NewIntOrStringFromInt(8080)},
				{Name: "port1", Port: 88, Protocol: "TCP", TargetPort: util.NewIntOrStringFromInt(8088)},
			},
		},
	})
	endpoints.syncService(ns + "/foo")
	expectedSubsets := []api.EndpointSubset{{
		Addresses: []api.EndpointAddress{
			{IP: "1.2.3.4", TargetRef: &api.ObjectReference{Kind: "Pod", Name: "pod0", Namespace: ns}},
			{IP: "1.2.3.5", TargetRef: &api.ObjectReference{Kind: "Pod", Name: "pod1", Namespace: ns}},
			{IP: "1.2.3.6", TargetRef: &api.ObjectReference{Kind: "Pod", Name: "pod2", Namespace: ns}},
		},
		Ports: []api.EndpointPort{
			{Name: "port0", Port: 8080, Protocol: "TCP"},
			{Name: "port1", Port: 8088, Protocol: "TCP"},
		},
	}}
	data := runtime.EncodeOrDie(testapi.Default.Codec(), &api.Endpoints{
		ObjectMeta: api.ObjectMeta{
			ResourceVersion: "",
			Labels:          serviceLabels,
		},
		Subsets: endptspkg.SortSubsets(expectedSubsets),
	})
	// endpointsHandler should get 2 requests - one for "GET" and the next for "POST".
	endpointsHandler.ValidateRequestCount(t, 2)
	endpointsHandler.ValidateRequest(t, testapi.Default.ResourcePath("endpoints", ns, ""), "POST", &data)
}

func TestSyncEndpointsItemsPreexistingLabelsChange(t *testing.T) {
	ns := "bar"
	testServer, endpointsHandler := makeTestServer(t, ns,
		serverResponse{http.StatusOK, &api.Endpoints{
			ObjectMeta: api.ObjectMeta{
				Name:            "foo",
				Namespace:       ns,
				ResourceVersion: "1",
				Labels: map[string]string{
					"foo": "bar",
				},
			},
			Subsets: []api.EndpointSubset{{
				Addresses: []api.EndpointAddress{{IP: "6.7.8.9"}},
				Ports:     []api.EndpointPort{{Port: 1000}},
			}},
		}})
	defer testServer.Close()
	client := client.NewOrDie(&client.Config{Host: testServer.URL, Version: testapi.Default.Version()})
	endpoints := NewEndpointController(client, controller.NoResyncPeriodFunc)
	addPods(endpoints.podStore.Store, ns, 1, 1, 0)
	serviceLabels := map[string]string{"baz": "blah"}
	endpoints.serviceStore.Store.Add(&api.Service{
		ObjectMeta: api.ObjectMeta{
			Name:      "foo",
			Namespace: ns,
			Labels:    serviceLabels,
		},
		Spec: api.ServiceSpec{
			Selector: map[string]string{"foo": "bar"},
			Ports:    []api.ServicePort{{Port: 80, Protocol: "TCP", TargetPort: util.NewIntOrStringFromInt(8080)}},
		},
	})
	endpoints.syncService(ns + "/foo")
	data := runtime.EncodeOrDie(testapi.Default.Codec(), &api.Endpoints{
		ObjectMeta: api.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
			Labels:          serviceLabels,
		},
		Subsets: []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{IP: "1.2.3.4", TargetRef: &api.ObjectReference{Kind: "Pod", Name: "pod0", Namespace: ns}}},
			Ports:     []api.EndpointPort{{Port: 8080, Protocol: "TCP"}},
		}},
	})
	endpointsHandler.ValidateRequest(t, testapi.Default.ResourcePath("endpoints", ns, "foo"), "PUT", &data)
}
