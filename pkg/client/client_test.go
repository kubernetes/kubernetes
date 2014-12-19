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

package client

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"path"
	"reflect"
	"strings"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/resources"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/version"
)

// TODO: Move this to a common place, it's needed in multiple tests.
const apiPath = "/api/v1beta1"
const nameRequiredError = "name is required parameter to Get"

type testRequest struct {
	Method  string
	Path    string
	Header  string
	Query   url.Values
	Body    runtime.Object
	RawBody *string
}

type Response struct {
	StatusCode int
	Body       runtime.Object
	RawBody    *string
}

type testClient struct {
	*Client
	Request  testRequest
	Response Response
	Error    bool
	Created  bool
	Version  string
	server   *httptest.Server
	handler  *util.FakeHandler
	// For query args, an optional function to validate the contents
	// useful when the contents can change but still be correct.
	// Maps from query arg key to validator.
	// If no validator is present, string equality is used.
	QueryValidator map[string]func(string, string) bool
}

func (c *testClient) Setup() *testClient {
	c.handler = &util.FakeHandler{
		StatusCode: c.Response.StatusCode,
	}
	if responseBody := body(c.Response.Body, c.Response.RawBody); responseBody != nil {
		c.handler.ResponseBody = *responseBody
	}
	c.server = httptest.NewServer(c.handler)
	if c.Client == nil {
		version := c.Version
		if len(version) == 0 {
			version = testapi.Version()
		}
		c.Client = NewOrDie(&Config{
			Host:    c.server.URL,
			Version: version,
		})
	}
	c.QueryValidator = map[string]func(string, string) bool{}
	return c
}

func (c *testClient) Validate(t *testing.T, received runtime.Object, err error) {
	c.ValidateCommon(t, err)

	if c.Response.Body != nil && !reflect.DeepEqual(c.Response.Body, received) {
		t.Errorf("bad response for request %#v: expected %#v, got %#v", c.Request, c.Response.Body, received)
	}
}

func (c *testClient) ValidateRaw(t *testing.T, received []byte, err error) {
	c.ValidateCommon(t, err)

	if c.Response.Body != nil && !reflect.DeepEqual(c.Response.Body, received) {
		t.Errorf("bad response for request %#v: expected %#v, got %#v", c.Request, c.Response.Body, received)
	}
}

func (c *testClient) ValidateCommon(t *testing.T, err error) {
	defer c.server.Close()

	if c.Error {
		if err == nil {
			t.Errorf("error expected for %#v, got none", c.Request)
		}
		return
	}
	if err != nil {
		t.Errorf("no error expected for %#v, got: %v", c.Request, err)
	}

	if c.handler.RequestReceived == nil {
		t.Errorf("handler had an empty request, %#v", c)
		return
	}

	requestBody := body(c.Request.Body, c.Request.RawBody)
	actualQuery := c.handler.RequestReceived.URL.Query()
	t.Logf("got query: %v", actualQuery)
	t.Logf("path: %v", c.Request.Path)
	// We check the query manually, so blank it out so that FakeHandler.ValidateRequest
	// won't check it.
	c.handler.RequestReceived.URL.RawQuery = ""
	c.handler.ValidateRequest(t, path.Join(apiPath, c.Request.Path), c.Request.Method, requestBody)
	for key, values := range c.Request.Query {
		validator, ok := c.QueryValidator[key]
		if !ok {
			switch key {
			case "labels", "fields":
				validator = validateLabels
			default:
				validator = func(a, b string) bool { return a == b }
			}
		}
		observed := actualQuery.Get(key)
		wanted := strings.Join(values, "")
		if !validator(wanted, observed) {
			t.Errorf("Unexpected query arg for key: %s.  Expected %s, Received %s", key, wanted, observed)
		}
	}
	if c.Request.Header != "" {
		if c.handler.RequestReceived.Header.Get(c.Request.Header) == "" {
			t.Errorf("header %q not found in request %#v", c.Request.Header, c.handler.RequestReceived)
		}
	}

	if expected, received := requestBody, c.handler.RequestBody; expected != nil && *expected != received {
		t.Errorf("bad body for request %#v: expected %s, got %s", c.Request, *expected, received)
	}
}

// buildResourcePath is a convenience function for knowing if a namespace should in a path param or not
func buildResourcePath(namespace, resource string) string {
	if len(namespace) > 0 {
		if NamespaceInPathFor(testapi.Version()) {
			return path.Join("ns", namespace, resource)
		}
	}
	return resource
}

// buildQueryValues is a convenience function for knowing if a namespace should go in a query param or not
func buildQueryValues(namespace string, query url.Values) url.Values {
	v := url.Values{}
	if query != nil {
		for key, values := range query {
			for _, value := range values {
				v.Add(key, value)
			}
		}
	}
	if len(namespace) > 0 {
		if !NamespaceInPathFor(testapi.Version()) {
			v.Set("namespace", namespace)
		}
	}
	return v
}

func TestListEmptyPods(t *testing.T) {
	ns := api.NamespaceDefault
	c := &testClient{
		Request:  testRequest{Method: "GET", Path: buildResourcePath(ns, "/pods"), Query: buildQueryValues(ns, nil)},
		Response: Response{StatusCode: 200, Body: &api.PodList{}},
	}
	podList, err := c.Setup().Pods(ns).List(labels.Everything())
	c.Validate(t, podList, err)
}

func TestListPods(t *testing.T) {
	ns := api.NamespaceDefault
	c := &testClient{
		Request: testRequest{Method: "GET", Path: buildResourcePath(ns, "/pods"), Query: buildQueryValues(ns, nil)},
		Response: Response{StatusCode: 200,
			Body: &api.PodList{
				Items: []api.Pod{
					{
						Status: api.PodStatus{
							Phase: api.PodRunning,
						},
						ObjectMeta: api.ObjectMeta{
							Labels: map[string]string{
								"foo":  "bar",
								"name": "baz",
							},
						},
					},
				},
			},
		},
	}
	receivedPodList, err := c.Setup().Pods(ns).List(labels.Everything())
	c.Validate(t, receivedPodList, err)
}

func validateLabels(a, b string) bool {
	sA, _ := labels.ParseSelector(a)
	sB, _ := labels.ParseSelector(b)
	return sA.String() == sB.String()
}

func TestListPodsLabels(t *testing.T) {
	ns := api.NamespaceDefault
	c := &testClient{
		Request: testRequest{Method: "GET", Path: buildResourcePath(ns, "/pods"), Query: buildQueryValues(ns, url.Values{"labels": []string{"foo=bar,name=baz"}})},
		Response: Response{
			StatusCode: 200,
			Body: &api.PodList{
				Items: []api.Pod{
					{
						Status: api.PodStatus{
							Phase: api.PodRunning,
						},
						ObjectMeta: api.ObjectMeta{
							Labels: map[string]string{
								"foo":  "bar",
								"name": "baz",
							},
						},
					},
				},
			},
		},
	}
	c.Setup()
	c.QueryValidator["labels"] = validateLabels
	selector := labels.Set{"foo": "bar", "name": "baz"}.AsSelector()
	receivedPodList, err := c.Pods(ns).List(selector)
	c.Validate(t, receivedPodList, err)
}

func TestGetPod(t *testing.T) {
	ns := api.NamespaceDefault
	c := &testClient{
		Request: testRequest{Method: "GET", Path: buildResourcePath(ns, "/pods/foo"), Query: buildQueryValues(ns, nil)},
		Response: Response{
			StatusCode: 200,
			Body: &api.Pod{
				Status: api.PodStatus{
					Phase: api.PodRunning,
				},
				ObjectMeta: api.ObjectMeta{
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
			},
		},
	}
	receivedPod, err := c.Setup().Pods(ns).Get("foo")
	c.Validate(t, receivedPod, err)
}

func TestGetPodWithNoName(t *testing.T) {
	ns := api.NamespaceDefault
	c := &testClient{Error: true}
	receivedPod, err := c.Setup().Pods(ns).Get("")
	if (err != nil) && (err.Error() != nameRequiredError) {
		t.Errorf("Expected error: %v, but got %v", nameRequiredError, err)
	}

	c.Validate(t, receivedPod, err)
}

func TestDeletePod(t *testing.T) {
	ns := api.NamespaceDefault
	c := &testClient{
		Request:  testRequest{Method: "DELETE", Path: buildResourcePath(ns, "/pods/foo"), Query: buildQueryValues(ns, nil)},
		Response: Response{StatusCode: 200},
	}
	err := c.Setup().Pods(ns).Delete("foo")
	c.Validate(t, nil, err)
}

func TestCreatePod(t *testing.T) {
	ns := api.NamespaceDefault
	requestPod := &api.Pod{
		Status: api.PodStatus{
			Phase: api.PodRunning,
		},
		ObjectMeta: api.ObjectMeta{
			Labels: map[string]string{
				"foo":  "bar",
				"name": "baz",
			},
		},
	}
	c := &testClient{
		Request: testRequest{Method: "POST", Path: buildResourcePath(ns, "/pods"), Query: buildQueryValues(ns, nil), Body: requestPod},
		Response: Response{
			StatusCode: 200,
			Body:       requestPod,
		},
	}
	receivedPod, err := c.Setup().Pods(ns).Create(requestPod)
	c.Validate(t, receivedPod, err)
}

func TestUpdatePod(t *testing.T) {
	ns := api.NamespaceDefault
	requestPod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:            "foo",
			ResourceVersion: "1",
			Labels: map[string]string{
				"foo":  "bar",
				"name": "baz",
			},
		},
		Status: api.PodStatus{
			Phase: api.PodRunning,
		},
	}
	c := &testClient{
		Request:  testRequest{Method: "PUT", Path: buildResourcePath(ns, "/pods/foo"), Query: buildQueryValues(ns, nil)},
		Response: Response{StatusCode: 200, Body: requestPod},
	}
	receivedPod, err := c.Setup().Pods(ns).Update(requestPod)
	c.Validate(t, receivedPod, err)
}

func TestListControllers(t *testing.T) {
	c := &testClient{
		Request: testRequest{Method: "GET", Path: "/replicationControllers"},
		Response: Response{StatusCode: 200,
			Body: &api.ReplicationControllerList{
				Items: []api.ReplicationController{
					{
						ObjectMeta: api.ObjectMeta{
							Name: "foo",
							Labels: map[string]string{
								"foo":  "bar",
								"name": "baz",
							},
						},
						Spec: api.ReplicationControllerSpec{
							Replicas: 2,
							Template: &api.PodTemplateSpec{},
						},
					},
				},
			},
		},
	}
	receivedControllerList, err := c.Setup().ReplicationControllers(api.NamespaceAll).List(labels.Everything())
	c.Validate(t, receivedControllerList, err)

}

func TestGetController(t *testing.T) {
	ns := api.NamespaceDefault
	c := &testClient{
		Request: testRequest{Method: "GET", Path: buildResourcePath(ns, "/replicationControllers/foo"), Query: buildQueryValues(ns, nil)},
		Response: Response{
			StatusCode: 200,
			Body: &api.ReplicationController{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Spec: api.ReplicationControllerSpec{
					Replicas: 2,
					Template: &api.PodTemplateSpec{},
				},
			},
		},
	}
	receivedController, err := c.Setup().ReplicationControllers(ns).Get("foo")
	c.Validate(t, receivedController, err)
}

func TestGetControllerWithNoName(t *testing.T) {
	ns := api.NamespaceDefault
	c := &testClient{Error: true}
	receivedPod, err := c.Setup().ReplicationControllers(ns).Get("")
	if (err != nil) && (err.Error() != nameRequiredError) {
		t.Errorf("Expected error: %v, but got %v", nameRequiredError, err)
	}

	c.Validate(t, receivedPod, err)
}

func TestUpdateController(t *testing.T) {
	ns := api.NamespaceDefault
	requestController := &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{Name: "foo", ResourceVersion: "1"},
	}
	c := &testClient{
		Request: testRequest{Method: "PUT", Path: buildResourcePath(ns, "/replicationControllers/foo"), Query: buildQueryValues(ns, nil)},
		Response: Response{
			StatusCode: 200,
			Body: &api.ReplicationController{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Spec: api.ReplicationControllerSpec{
					Replicas: 2,
					Template: &api.PodTemplateSpec{},
				},
			},
		},
	}
	receivedController, err := c.Setup().ReplicationControllers(ns).Update(requestController)
	c.Validate(t, receivedController, err)
}

func TestDeleteController(t *testing.T) {
	ns := api.NamespaceDefault
	c := &testClient{
		Request:  testRequest{Method: "DELETE", Path: buildResourcePath(ns, "/replicationControllers/foo"), Query: buildQueryValues(ns, nil)},
		Response: Response{StatusCode: 200},
	}
	err := c.Setup().ReplicationControllers(ns).Delete("foo")
	c.Validate(t, nil, err)
}

func TestCreateController(t *testing.T) {
	ns := api.NamespaceDefault
	requestController := &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
	}
	c := &testClient{
		Request: testRequest{Method: "POST", Path: buildResourcePath(ns, "/replicationControllers"), Body: requestController, Query: buildQueryValues(ns, nil)},
		Response: Response{
			StatusCode: 200,
			Body: &api.ReplicationController{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Spec: api.ReplicationControllerSpec{
					Replicas: 2,
					Template: &api.PodTemplateSpec{},
				},
			},
		},
	}
	receivedController, err := c.Setup().ReplicationControllers(ns).Create(requestController)
	c.Validate(t, receivedController, err)
}

func body(obj runtime.Object, raw *string) *string {
	if obj != nil {
		bs, _ := latest.Codec.Encode(obj)
		body := string(bs)
		return &body
	}
	return raw
}

func TestListServices(t *testing.T) {
	ns := api.NamespaceDefault
	c := &testClient{
		Request: testRequest{Method: "GET", Path: buildResourcePath(ns, "/services"), Query: buildQueryValues(ns, nil)},
		Response: Response{StatusCode: 200,
			Body: &api.ServiceList{
				Items: []api.Service{
					{
						ObjectMeta: api.ObjectMeta{
							Name: "name",
							Labels: map[string]string{
								"foo":  "bar",
								"name": "baz",
							},
						},
						Spec: api.ServiceSpec{
							Selector: map[string]string{
								"one": "two",
							},
						},
					},
				},
			},
		},
	}
	receivedServiceList, err := c.Setup().Services(ns).List(labels.Everything())
	t.Logf("received services: %v %#v", err, receivedServiceList)
	c.Validate(t, receivedServiceList, err)
}

func TestListServicesLabels(t *testing.T) {
	ns := api.NamespaceDefault
	c := &testClient{
		Request: testRequest{Method: "GET", Path: buildResourcePath(ns, "/services"), Query: buildQueryValues(ns, url.Values{"labels": []string{"foo=bar,name=baz"}})},
		Response: Response{StatusCode: 200,
			Body: &api.ServiceList{
				Items: []api.Service{
					{
						ObjectMeta: api.ObjectMeta{
							Name: "name",
							Labels: map[string]string{
								"foo":  "bar",
								"name": "baz",
							},
						},
						Spec: api.ServiceSpec{
							Selector: map[string]string{
								"one": "two",
							},
						},
					},
				},
			},
		},
	}
	c.Setup()
	c.QueryValidator["labels"] = validateLabels
	selector := labels.Set{"foo": "bar", "name": "baz"}.AsSelector()
	receivedServiceList, err := c.Services(ns).List(selector)
	c.Validate(t, receivedServiceList, err)
}

func TestGetService(t *testing.T) {
	ns := api.NamespaceDefault
	c := &testClient{
		Request:  testRequest{Method: "GET", Path: buildResourcePath(ns, "/services/1"), Query: buildQueryValues(ns, nil)},
		Response: Response{StatusCode: 200, Body: &api.Service{ObjectMeta: api.ObjectMeta{Name: "service-1"}}},
	}
	response, err := c.Setup().Services(ns).Get("1")
	c.Validate(t, response, err)
}

func TestGetServiceWithNoName(t *testing.T) {
	ns := api.NamespaceDefault
	c := &testClient{Error: true}
	receivedPod, err := c.Setup().Services(ns).Get("")
	if (err != nil) && (err.Error() != nameRequiredError) {
		t.Errorf("Expected error: %v, but got %v", nameRequiredError, err)
	}

	c.Validate(t, receivedPod, err)
}

func TestCreateService(t *testing.T) {
	ns := api.NamespaceDefault
	c := &testClient{
		Request:  testRequest{Method: "POST", Path: buildResourcePath(ns, "/services"), Body: &api.Service{ObjectMeta: api.ObjectMeta{Name: "service-1"}}, Query: buildQueryValues(ns, nil)},
		Response: Response{StatusCode: 200, Body: &api.Service{ObjectMeta: api.ObjectMeta{Name: "service-1"}}},
	}
	response, err := c.Setup().Services(ns).Create(&api.Service{ObjectMeta: api.ObjectMeta{Name: "service-1"}})
	c.Validate(t, response, err)
}

func TestUpdateService(t *testing.T) {
	ns := api.NamespaceDefault
	svc := &api.Service{ObjectMeta: api.ObjectMeta{Name: "service-1", ResourceVersion: "1"}}
	c := &testClient{
		Request:  testRequest{Method: "PUT", Path: buildResourcePath(ns, "/services/service-1"), Body: svc, Query: buildQueryValues(ns, nil)},
		Response: Response{StatusCode: 200, Body: svc},
	}
	response, err := c.Setup().Services(ns).Update(svc)
	c.Validate(t, response, err)
}

func TestDeleteService(t *testing.T) {
	ns := api.NamespaceDefault
	c := &testClient{
		Request:  testRequest{Method: "DELETE", Path: buildResourcePath(ns, "/services/1"), Query: buildQueryValues(ns, nil)},
		Response: Response{StatusCode: 200},
	}
	err := c.Setup().Services(ns).Delete("1")
	c.Validate(t, nil, err)
}

func TestListEndpooints(t *testing.T) {
	ns := api.NamespaceDefault
	c := &testClient{
		Request: testRequest{Method: "GET", Path: buildResourcePath(ns, "/endpoints"), Query: buildQueryValues(ns, nil)},
		Response: Response{StatusCode: 200,
			Body: &api.EndpointsList{
				Items: []api.Endpoints{
					{
						ObjectMeta: api.ObjectMeta{Name: "endpoint-1"},
						Endpoints:  []string{"10.245.1.2:8080", "10.245.1.3:8080"},
					},
				},
			},
		},
	}
	receivedEndpointsList, err := c.Setup().Endpoints(ns).List(labels.Everything())
	c.Validate(t, receivedEndpointsList, err)
}

func TestGetEndpoints(t *testing.T) {
	ns := api.NamespaceDefault
	c := &testClient{
		Request:  testRequest{Method: "GET", Path: buildResourcePath(ns, "/endpoints/endpoint-1"), Query: buildQueryValues(ns, nil)},
		Response: Response{StatusCode: 200, Body: &api.Endpoints{ObjectMeta: api.ObjectMeta{Name: "endpoint-1"}}},
	}
	response, err := c.Setup().Endpoints(ns).Get("endpoint-1")
	c.Validate(t, response, err)
}

func TestGetEndpointWithNoName(t *testing.T) {
	ns := api.NamespaceDefault
	c := &testClient{Error: true}
	receivedPod, err := c.Setup().Endpoints(ns).Get("")
	if (err != nil) && (err.Error() != nameRequiredError) {
		t.Errorf("Expected error: %v, but got %v", nameRequiredError, err)
	}

	c.Validate(t, receivedPod, err)
}

func TestGetServerVersion(t *testing.T) {
	expect := version.Info{
		Major:     "foo",
		Minor:     "bar",
		GitCommit: "baz",
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		output, err := json.Marshal(expect)
		if err != nil {
			t.Errorf("unexpected encoding error: %v", err)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write(output)
	}))
	client := NewOrDie(&Config{Host: server.URL})

	got, err := client.ServerVersion()
	if err != nil {
		t.Fatalf("unexpected encoding error: %v", err)
	}
	if e, a := expect, *got; !reflect.DeepEqual(e, a) {
		t.Errorf("expected %v, got %v", e, a)
	}
}

func TestGetServerAPIVersions(t *testing.T) {
	versions := []string{"v1", "v2", "v3"}
	expect := api.APIVersions{Versions: versions}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		output, err := json.Marshal(expect)
		if err != nil {
			t.Errorf("unexpected encoding error: %v", err)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write(output)
	}))
	client := NewOrDie(&Config{Host: server.URL})
	got, err := client.ServerAPIVersions()
	if err != nil {
		t.Fatalf("unexpected encoding error: %v", err)
	}
	if e, a := expect, *got; !reflect.DeepEqual(e, a) {
		t.Errorf("expected %v, got %v", e, a)
	}
}

func TestListMinions(t *testing.T) {
	c := &testClient{
		Request:  testRequest{Method: "GET", Path: "/minions"},
		Response: Response{StatusCode: 200, Body: &api.NodeList{ListMeta: api.ListMeta{ResourceVersion: "1"}}},
	}
	response, err := c.Setup().Nodes().List()
	c.Validate(t, response, err)
}

func TestGetMinion(t *testing.T) {
	c := &testClient{
		Request:  testRequest{Method: "GET", Path: "/minions/1"},
		Response: Response{StatusCode: 200, Body: &api.Node{ObjectMeta: api.ObjectMeta{Name: "minion-1"}}},
	}
	response, err := c.Setup().Nodes().Get("1")
	c.Validate(t, response, err)
}

func TestGetMinionWithNoName(t *testing.T) {
	c := &testClient{Error: true}
	receivedPod, err := c.Setup().Nodes().Get("")
	if (err != nil) && (err.Error() != nameRequiredError) {
		t.Errorf("Expected error: %v, but got %v", nameRequiredError, err)
	}

	c.Validate(t, receivedPod, err)
}

func TestCreateMinion(t *testing.T) {
	requestMinion := &api.Node{
		ObjectMeta: api.ObjectMeta{
			Name: "minion-1",
		},
		Status: api.NodeStatus{
			HostIP: "123.321.456.654",
		},
		Spec: api.NodeSpec{
			Capacity: api.ResourceList{
				resources.CPU:    util.NewIntOrStringFromInt(1000),
				resources.Memory: util.NewIntOrStringFromInt(1024 * 1024),
			},
		},
	}
	c := &testClient{
		Request: testRequest{Method: "POST", Path: "/minions", Body: requestMinion},
		Response: Response{
			StatusCode: 200,
			Body:       requestMinion,
		},
	}
	receivedMinion, err := c.Setup().Nodes().Create(requestMinion)
	c.Validate(t, receivedMinion, err)
}

func TestDeleteMinion(t *testing.T) {
	c := &testClient{
		Request:  testRequest{Method: "DELETE", Path: "/minions/foo"},
		Response: Response{StatusCode: 200},
	}
	err := c.Setup().Nodes().Delete("foo")
	c.Validate(t, nil, err)
}

func TestNewMinionPath(t *testing.T) {
	c := &testClient{
		Request:  testRequest{Method: "DELETE", Path: "/nodes/foo"},
		Response: Response{StatusCode: 200},
	}
	cl := c.Setup()
	cl.preV1Beta3 = false
	err := cl.Nodes().Delete("foo")
	c.Validate(t, nil, err)
}
