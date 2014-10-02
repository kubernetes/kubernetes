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
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/version"
)

// TODO: Move this to a common place, it's needed in multiple tests.
const apiPath = "/api/v1beta1"

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
		c.Client = NewOrDie(&Config{
			Host:    c.server.URL,
			Version: "v1beta1",
		})
	}
	c.QueryValidator = map[string]func(string, string) bool{}
	return c
}

func (c *testClient) Validate(t *testing.T, received runtime.Object, err error) {
	c.ValidateCommon(t, err)

	if c.Response.Body != nil && !reflect.DeepEqual(c.Response.Body, received) {
		t.Errorf("bad response for request %#v: expected %s, got %s", c.Request, c.Response.Body, received)
	}
}

func (c *testClient) ValidateRaw(t *testing.T, received []byte, err error) {
	c.ValidateCommon(t, err)

	if c.Response.Body != nil && !reflect.DeepEqual(c.Response.Body, received) {
		t.Errorf("bad response for request %#v: expected %s, got %s", c.Request, c.Response.Body, received)
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
	// We check the query manually, so blank it out so that FakeHandler.ValidateRequest
	// won't check it.
	c.handler.RequestReceived.URL.RawQuery = ""
	c.handler.ValidateRequest(t, path.Join(apiPath, c.Request.Path), c.Request.Method, requestBody)
	for key, values := range c.Request.Query {
		validator, ok := c.QueryValidator[key]
		if !ok {
			validator = func(a, b string) bool { return a == b }
		}
		observed := actualQuery.Get(key)
		if !validator(values[0], observed) {
			t.Errorf("Unexpected query arg for key: %s.  Expected %s, Received %s", key, values[0], observed)
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

func TestListEmptyPods(t *testing.T) {
	ctx := api.NewContext()
	c := &testClient{
		Request:  testRequest{Method: "GET", Path: "/pods"},
		Response: Response{StatusCode: 200, Body: &api.PodList{}},
	}
	podList, err := c.Setup().ListPods(ctx, labels.Everything())
	c.Validate(t, podList, err)
}

func TestListPods(t *testing.T) {
	ctx := api.NewDefaultContext()
	c := &testClient{
		Request: testRequest{Method: "GET", Path: "/pods"},
		Response: Response{StatusCode: 200,
			Body: &api.PodList{
				Items: []api.Pod{
					{
						CurrentState: api.PodState{
							Status: "Foobar",
						},
						Labels: map[string]string{
							"foo":  "bar",
							"name": "baz",
						},
					},
				},
			},
		},
	}
	receivedPodList, err := c.Setup().ListPods(ctx, labels.Everything())
	c.Validate(t, receivedPodList, err)
}

func validateLabels(a, b string) bool {
	sA, _ := labels.ParseSelector(a)
	sB, _ := labels.ParseSelector(b)
	return sA.String() == sB.String()
}

func TestListPodsLabels(t *testing.T) {
	ctx := api.NewDefaultContext()
	c := &testClient{
		Request: testRequest{Method: "GET", Path: "/pods", Query: url.Values{"labels": []string{"foo=bar,name=baz"}}},
		Response: Response{
			StatusCode: 200,
			Body: &api.PodList{
				Items: []api.Pod{
					{
						CurrentState: api.PodState{
							Status: "Foobar",
						},
						Labels: map[string]string{
							"foo":  "bar",
							"name": "baz",
						},
					},
				},
			},
		},
	}
	c.Setup()
	c.QueryValidator["labels"] = validateLabels
	selector := labels.Set{"foo": "bar", "name": "baz"}.AsSelector()
	receivedPodList, err := c.ListPods(ctx, selector)
	c.Validate(t, receivedPodList, err)
}

func TestGetPod(t *testing.T) {
	ctx := api.NewDefaultContext()
	c := &testClient{
		Request: testRequest{Method: "GET", Path: "/pods/foo"},
		Response: Response{
			StatusCode: 200,
			Body: &api.Pod{
				CurrentState: api.PodState{
					Status: "Foobar",
				},
				Labels: map[string]string{
					"foo":  "bar",
					"name": "baz",
				},
			},
		},
	}
	receivedPod, err := c.Setup().GetPod(ctx, "foo")
	c.Validate(t, receivedPod, err)
}

func TestDeletePod(t *testing.T) {
	c := &testClient{
		Request:  testRequest{Method: "DELETE", Path: "/pods/foo"},
		Response: Response{StatusCode: 200},
	}
	err := c.Setup().DeletePod(api.NewDefaultContext(), "foo")
	c.Validate(t, nil, err)
}

func TestCreatePod(t *testing.T) {
	requestPod := &api.Pod{
		CurrentState: api.PodState{
			Status: "Foobar",
		},
		Labels: map[string]string{
			"foo":  "bar",
			"name": "baz",
		},
	}
	c := &testClient{
		Request: testRequest{Method: "POST", Path: "/pods", Body: requestPod},
		Response: Response{
			StatusCode: 200,
			Body:       requestPod,
		},
	}
	receivedPod, err := c.Setup().CreatePod(api.NewDefaultContext(), requestPod)
	c.Validate(t, receivedPod, err)
}

func TestUpdatePod(t *testing.T) {
	requestPod := &api.Pod{
		JSONBase: api.JSONBase{ID: "foo", ResourceVersion: 1},
		CurrentState: api.PodState{
			Status: "Foobar",
		},
		Labels: map[string]string{
			"foo":  "bar",
			"name": "baz",
		},
	}
	c := &testClient{
		Request:  testRequest{Method: "PUT", Path: "/pods/foo"},
		Response: Response{StatusCode: 200, Body: requestPod},
	}
	receivedPod, err := c.Setup().UpdatePod(api.NewDefaultContext(), requestPod)
	c.Validate(t, receivedPod, err)
}

func TestListControllers(t *testing.T) {
	c := &testClient{
		Request: testRequest{Method: "GET", Path: "/replicationControllers"},
		Response: Response{StatusCode: 200,
			Body: &api.ReplicationControllerList{
				Items: []api.ReplicationController{
					{
						JSONBase: api.JSONBase{ID: "foo"},
						DesiredState: api.ReplicationControllerState{
							Replicas: 2,
						},
						Labels: map[string]string{
							"foo":  "bar",
							"name": "baz",
						},
					},
				},
			},
		},
	}
	receivedControllerList, err := c.Setup().ListReplicationControllers(api.NewContext(), labels.Everything())
	c.Validate(t, receivedControllerList, err)

}

func TestGetController(t *testing.T) {
	c := &testClient{
		Request: testRequest{Method: "GET", Path: "/replicationControllers/foo"},
		Response: Response{
			StatusCode: 200,
			Body: &api.ReplicationController{
				JSONBase: api.JSONBase{ID: "foo"},
				DesiredState: api.ReplicationControllerState{
					Replicas: 2,
				},
				Labels: map[string]string{
					"foo":  "bar",
					"name": "baz",
				},
			},
		},
	}
	receivedController, err := c.Setup().GetReplicationController(api.NewDefaultContext(), "foo")
	c.Validate(t, receivedController, err)
}

func TestUpdateController(t *testing.T) {
	requestController := &api.ReplicationController{
		JSONBase: api.JSONBase{ID: "foo", ResourceVersion: 1},
	}
	c := &testClient{
		Request: testRequest{Method: "PUT", Path: "/replicationControllers/foo"},
		Response: Response{
			StatusCode: 200,
			Body: &api.ReplicationController{
				JSONBase: api.JSONBase{ID: "foo"},
				DesiredState: api.ReplicationControllerState{
					Replicas: 2,
				},
				Labels: map[string]string{
					"foo":  "bar",
					"name": "baz",
				},
			},
		},
	}
	receivedController, err := c.Setup().UpdateReplicationController(api.NewDefaultContext(), requestController)
	c.Validate(t, receivedController, err)
}

func TestDeleteController(t *testing.T) {
	c := &testClient{
		Request:  testRequest{Method: "DELETE", Path: "/replicationControllers/foo"},
		Response: Response{StatusCode: 200},
	}
	err := c.Setup().DeleteReplicationController(api.NewDefaultContext(), "foo")
	c.Validate(t, nil, err)
}

func TestCreateController(t *testing.T) {
	requestController := &api.ReplicationController{
		JSONBase: api.JSONBase{ID: "foo"},
	}
	c := &testClient{
		Request: testRequest{Method: "POST", Path: "/replicationControllers", Body: requestController},
		Response: Response{
			StatusCode: 200,
			Body: &api.ReplicationController{
				JSONBase: api.JSONBase{ID: "foo"},
				DesiredState: api.ReplicationControllerState{
					Replicas: 2,
				},
				Labels: map[string]string{
					"foo":  "bar",
					"name": "baz",
				},
			},
		},
	}
	receivedController, err := c.Setup().CreateReplicationController(api.NewDefaultContext(), requestController)
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
	c := &testClient{
		Request: testRequest{Method: "GET", Path: "/services"},
		Response: Response{StatusCode: 200,
			Body: &api.ServiceList{
				Items: []api.Service{
					{
						JSONBase: api.JSONBase{ID: "name"},
						Labels: map[string]string{
							"foo":  "bar",
							"name": "baz",
						},
						Selector: map[string]string{
							"one": "two",
						},
					},
				},
			},
		},
	}
	receivedServiceList, err := c.Setup().ListServices(api.NewDefaultContext(), labels.Everything())
	c.Validate(t, receivedServiceList, err)
}

func TestListServicesLabels(t *testing.T) {
	c := &testClient{
		Request: testRequest{Method: "GET", Path: "/services", Query: url.Values{"labels": []string{"foo=bar,name=baz"}}},
		Response: Response{StatusCode: 200,
			Body: &api.ServiceList{
				Items: []api.Service{
					{
						JSONBase: api.JSONBase{ID: "name"},
						Labels: map[string]string{
							"foo":  "bar",
							"name": "baz",
						},
						Selector: map[string]string{
							"one": "two",
						},
					},
				},
			},
		},
	}
	c.Setup()
	c.QueryValidator["labels"] = validateLabels
	selector := labels.Set{"foo": "bar", "name": "baz"}.AsSelector()
	receivedServiceList, err := c.ListServices(api.NewDefaultContext(), selector)
	c.Validate(t, receivedServiceList, err)
}

func TestGetService(t *testing.T) {
	c := &testClient{
		Request:  testRequest{Method: "GET", Path: "/services/1"},
		Response: Response{StatusCode: 200, Body: &api.Service{JSONBase: api.JSONBase{ID: "service-1"}}},
	}
	response, err := c.Setup().GetService(api.NewDefaultContext(), "1")
	c.Validate(t, response, err)
}

func TestCreateService(t *testing.T) {
	c := &testClient{
		Request:  testRequest{Method: "POST", Path: "/services", Body: &api.Service{JSONBase: api.JSONBase{ID: "service-1"}}},
		Response: Response{StatusCode: 200, Body: &api.Service{JSONBase: api.JSONBase{ID: "service-1"}}},
	}
	response, err := c.Setup().CreateService(api.NewDefaultContext(), &api.Service{JSONBase: api.JSONBase{ID: "service-1"}})
	c.Validate(t, response, err)
}

func TestUpdateService(t *testing.T) {
	svc := &api.Service{JSONBase: api.JSONBase{ID: "service-1", ResourceVersion: 1}}
	c := &testClient{
		Request:  testRequest{Method: "PUT", Path: "/services/service-1", Body: svc},
		Response: Response{StatusCode: 200, Body: svc},
	}
	response, err := c.Setup().UpdateService(api.NewDefaultContext(), svc)
	c.Validate(t, response, err)
}

func TestDeleteService(t *testing.T) {
	c := &testClient{
		Request:  testRequest{Method: "DELETE", Path: "/services/1"},
		Response: Response{StatusCode: 200},
	}
	err := c.Setup().DeleteService(api.NewDefaultContext(), "1")
	c.Validate(t, nil, err)
}

func TestListEndpooints(t *testing.T) {
	c := &testClient{
		Request: testRequest{Method: "GET", Path: "/endpoints"},
		Response: Response{StatusCode: 200,
			Body: &api.EndpointsList{
				Items: []api.Endpoints{
					{
						JSONBase:  api.JSONBase{ID: "endpoint-1"},
						Endpoints: []string{"10.245.1.2:8080", "10.245.1.3:8080"},
					},
				},
			},
		},
	}
	receivedEndpointsList, err := c.Setup().ListEndpoints(api.NewDefaultContext(), labels.Everything())
	c.Validate(t, receivedEndpointsList, err)
}

func TestGetEndpoints(t *testing.T) {
	c := &testClient{
		Request:  testRequest{Method: "GET", Path: "/endpoints/endpoint-1"},
		Response: Response{StatusCode: 200, Body: &api.Endpoints{JSONBase: api.JSONBase{ID: "endpoint-1"}}},
	}
	response, err := c.Setup().GetEndpoints(api.NewDefaultContext(), "endpoint-1")
	c.Validate(t, response, err)
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

func TestListMinions(t *testing.T) {
	c := &testClient{
		Request:  testRequest{Method: "GET", Path: "/minions"},
		Response: Response{StatusCode: 200, Body: &api.MinionList{JSONBase: api.JSONBase{ID: "minion-1"}}},
	}
	response, err := c.Setup().ListMinions()
	c.Validate(t, response, err)
}
