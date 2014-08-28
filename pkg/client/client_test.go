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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/version"
)

// TODO: Move this to a common place, it's needed in multiple tests.
const apiPath = "/api/v1beta1"

func TestValidatesHostParameter(t *testing.T) {
	testCases := map[string]struct {
		Host   string
		Prefix string
		Err    bool
	}{
		"127.0.0.1":          {"http://127.0.0.1", "/api/v1beta1/", false},
		"127.0.0.1:8080":     {"http://127.0.0.1:8080", "/api/v1beta1/", false},
		"foo.bar.com":        {"http://foo.bar.com", "/api/v1beta1/", false},
		"http://host/server": {"http://host", "/server/api/v1beta1/", false},
		"host/server":        {"", "", true},
	}
	for k, expected := range testCases {
		c, err := NewRESTClient(k, nil, "/api/v1beta1/")
		switch {
		case err == nil && expected.Err:
			t.Errorf("expected error but was nil")
			continue
		case err != nil && !expected.Err:
			t.Errorf("unexpected error %v", err)
			continue
		case err != nil:
			continue
		}
		if e, a := expected.Host, c.host; e != a {
			t.Errorf("%s: expected host %s, got %s", k, e, a)
			continue
		}
		if e, a := expected.Prefix, c.prefix; e != a {
			t.Errorf("%s: expected prefix %s, got %s", k, e, a)
			continue
		}
	}
}

func TestListEmptyPods(t *testing.T) {
	c := &testClient{
		Request:  testRequest{Method: "GET", Path: "/pods"},
		Response: Response{StatusCode: 200, Body: api.PodList{}},
	}
	podList, err := c.Setup().ListPods(labels.Everything())
	c.Validate(t, podList, err)
}

func TestListPods(t *testing.T) {
	c := &testClient{
		Request: testRequest{Method: "GET", Path: "/pods"},
		Response: Response{StatusCode: 200,
			Body: api.PodList{
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
	receivedPodList, err := c.Setup().ListPods(labels.Everything())
	c.Validate(t, receivedPodList, err)
}

func validateLabels(a, b string) bool {
	sA, _ := labels.ParseSelector(a)
	sB, _ := labels.ParseSelector(b)
	return sA.String() == sB.String()
}

func TestListPodsLabels(t *testing.T) {
	c := &testClient{
		Request: testRequest{Method: "GET", Path: "/pods", Query: url.Values{"labels": []string{"foo=bar,name=baz"}}},
		Response: Response{
			StatusCode: 200,
			Body: api.PodList{
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
	receivedPodList, err := c.ListPods(selector)
	c.Validate(t, receivedPodList, err)
}

func TestGetPod(t *testing.T) {
	c := &testClient{
		Request: testRequest{Method: "GET", Path: "/pods/foo"},
		Response: Response{
			StatusCode: 200,
			Body: api.Pod{
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
	receivedPod, err := c.Setup().GetPod("foo")
	c.Validate(t, receivedPod, err)
}

func TestDeletePod(t *testing.T) {
	c := &testClient{
		Request:  testRequest{Method: "DELETE", Path: "/pods/foo"},
		Response: Response{StatusCode: 200},
	}
	err := c.Setup().DeletePod("foo")
	c.Validate(t, nil, err)
}

func TestCreatePod(t *testing.T) {
	requestPod := api.Pod{
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
	receivedPod, err := c.Setup().CreatePod(requestPod)
	c.Validate(t, receivedPod, err)
}

func TestUpdatePod(t *testing.T) {
	requestPod := api.Pod{
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
	receivedPod, err := c.Setup().UpdatePod(requestPod)
	c.Validate(t, receivedPod, err)
}

func TestListControllers(t *testing.T) {
	c := &testClient{
		Request: testRequest{Method: "GET", Path: "/replicationControllers"},
		Response: Response{StatusCode: 200,
			Body: api.ReplicationControllerList{
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
	receivedControllerList, err := c.Setup().ListReplicationControllers(labels.Everything())
	c.Validate(t, receivedControllerList, err)

}

func TestGetController(t *testing.T) {
	c := &testClient{
		Request: testRequest{Method: "GET", Path: "/replicationControllers/foo"},
		Response: Response{
			StatusCode: 200,
			Body: api.ReplicationController{
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
	receivedController, err := c.Setup().GetReplicationController("foo")
	c.Validate(t, receivedController, err)
}

func TestUpdateController(t *testing.T) {
	requestController := api.ReplicationController{
		JSONBase: api.JSONBase{ID: "foo", ResourceVersion: 1},
	}
	c := &testClient{
		Request: testRequest{Method: "PUT", Path: "/replicationControllers/foo"},
		Response: Response{
			StatusCode: 200,
			Body: api.ReplicationController{
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
	receivedController, err := c.Setup().UpdateReplicationController(requestController)
	c.Validate(t, receivedController, err)
}

func TestDeleteController(t *testing.T) {
	c := &testClient{
		Request:  testRequest{Method: "DELETE", Path: "/replicationControllers/foo"},
		Response: Response{StatusCode: 200},
	}
	err := c.Setup().DeleteReplicationController("foo")
	c.Validate(t, nil, err)
}

func TestCreateController(t *testing.T) {
	requestController := api.ReplicationController{
		JSONBase: api.JSONBase{ID: "foo"},
	}
	c := &testClient{
		Request: testRequest{Method: "POST", Path: "/replicationControllers", Body: requestController},
		Response: Response{
			StatusCode: 200,
			Body: api.ReplicationController{
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
	receivedController, err := c.Setup().CreateReplicationController(requestController)
	c.Validate(t, receivedController, err)
}

func body(obj interface{}, raw *string) *string {
	if obj != nil {
		bs, _ := api.Encode(obj)
		body := string(bs)
		return &body
	}
	return raw
}

type testRequest struct {
	Method  string
	Path    string
	Header  string
	Query   url.Values
	Body    interface{}
	RawBody *string
}

type Response struct {
	StatusCode int
	Body       interface{}
	RawBody    *string
}

type testClient struct {
	*Client
	Request  testRequest
	Response Response
	Error    bool
	server   *httptest.Server
	handler  *util.FakeHandler
	Target   interface{}
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
		c.Client = NewOrDie("localhost", nil)
	}
	c.Client.host = c.server.URL
	c.Client.prefix = "/api/v1beta1/"
	c.QueryValidator = map[string]func(string, string) bool{}
	return c
}

func (c *testClient) Validate(t *testing.T, received interface{}, err error) {
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

	if c.Response.Body != nil && !reflect.DeepEqual(c.Response.Body, received) {
		t.Errorf("bad response for request %#v: expected %s, got %s", c.Request, c.Response.Body, received)
	}
}

func TestGetService(t *testing.T) {
	c := &testClient{
		Request:  testRequest{Method: "GET", Path: "/services/1"},
		Response: Response{StatusCode: 200, Body: &api.Service{JSONBase: api.JSONBase{ID: "service-1"}}},
	}
	response, err := c.Setup().GetService("1")
	c.Validate(t, &response, err)
}

func TestCreateService(t *testing.T) {
	c := (&testClient{
		Request:  testRequest{Method: "POST", Path: "/services", Body: &api.Service{JSONBase: api.JSONBase{ID: "service-1"}}},
		Response: Response{StatusCode: 200, Body: &api.Service{JSONBase: api.JSONBase{ID: "service-1"}}},
	}).Setup()
	response, err := c.Setup().CreateService(api.Service{JSONBase: api.JSONBase{ID: "service-1"}})
	c.Validate(t, &response, err)
}

func TestUpdateService(t *testing.T) {
	svc := api.Service{JSONBase: api.JSONBase{ID: "service-1", ResourceVersion: 1}}
	c := &testClient{
		Request:  testRequest{Method: "PUT", Path: "/services/service-1", Body: &svc},
		Response: Response{StatusCode: 200, Body: &svc},
	}
	response, err := c.Setup().UpdateService(svc)
	c.Validate(t, &response, err)
}

func TestDeleteService(t *testing.T) {
	c := &testClient{
		Request:  testRequest{Method: "DELETE", Path: "/services/1"},
		Response: Response{StatusCode: 200},
	}
	err := c.Setup().DeleteService("1")
	c.Validate(t, nil, err)
}

func TestDoRequest(t *testing.T) {
	invalid := "aaaaa"
	testClients := []testClient{
		{Request: testRequest{Method: "GET", Path: "good"}, Response: Response{StatusCode: 200}},
		{Request: testRequest{Method: "GET", Path: "bad%ZZ"}, Error: true},
		{Client: NewOrDie("localhost", &AuthInfo{"foo", "bar"}), Request: testRequest{Method: "GET", Path: "auth", Header: "Authorization"}, Response: Response{StatusCode: 200}},
		{Client: &Client{&RESTClient{httpClient: http.DefaultClient}}, Request: testRequest{Method: "GET", Path: "nocertificate"}, Error: true},
		{Request: testRequest{Method: "GET", Path: "error"}, Response: Response{StatusCode: 500}, Error: true},
		{Request: testRequest{Method: "POST", Path: "faildecode"}, Response: Response{StatusCode: 200, RawBody: &invalid}, Target: &struct{}{}},
		{Request: testRequest{Method: "GET", Path: "failread"}, Response: Response{StatusCode: 200, RawBody: &invalid}, Target: &struct{}{}},
	}
	for _, c := range testClients {
		client := c.Setup()
		prefix, _ := url.Parse(client.host)
		prefix.Path = client.prefix + c.Request.Path
		request := &http.Request{
			Method: c.Request.Method,
			Header: make(http.Header),
			URL:    prefix,
		}
		response, err := client.doRequest(request)
		c.Validate(t, response, err)
	}
}

func TestDoRequestAccepted(t *testing.T) {
	status := api.Status{Status: api.StatusWorking}
	expectedBody, _ := api.Encode(status)
	fakeHandler := util.FakeHandler{
		StatusCode:   202,
		ResponseBody: string(expectedBody),
		T:            t,
	}
	testServer := httptest.NewServer(&fakeHandler)
	request, _ := http.NewRequest("GET", testServer.URL+"/foo/bar", nil)
	auth := AuthInfo{User: "user", Password: "pass"}
	c, err := New(testServer.URL, &auth)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	body, err := c.doRequest(request)
	if request.Header["Authorization"] == nil {
		t.Errorf("Request is missing authorization header: %#v", *request)
	}
	if err == nil {
		t.Error("Unexpected non-error")
		return
	}
	se, ok := err.(*StatusErr)
	if !ok {
		t.Errorf("Unexpected kind of error: %#v", err)
		return
	}
	if !reflect.DeepEqual(se.Status, status) {
		t.Errorf("Unexpected status: %#v", se.Status)
	}
	if body != nil {
		t.Errorf("Expected nil body, but saw: '%s'", body)
	}
	fakeHandler.ValidateRequest(t, "/foo/bar", "GET", nil)
}

func TestDoRequestAcceptedSuccess(t *testing.T) {
	status := api.Status{Status: api.StatusSuccess}
	expectedBody, _ := api.Encode(status)
	fakeHandler := util.FakeHandler{
		StatusCode:   202,
		ResponseBody: string(expectedBody),
		T:            t,
	}
	testServer := httptest.NewServer(&fakeHandler)
	request, _ := http.NewRequest("GET", testServer.URL+"/foo/bar", nil)
	auth := AuthInfo{User: "user", Password: "pass"}
	c, err := New(testServer.URL, &auth)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	body, err := c.doRequest(request)
	if request.Header["Authorization"] == nil {
		t.Errorf("Request is missing authorization header: %#v", *request)
	}
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	statusOut, err := api.Decode(body)
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	if !reflect.DeepEqual(&status, statusOut) {
		t.Errorf("Unexpected mis-match. Expected %#v.  Saw %#v", status, statusOut)
	}
	fakeHandler.ValidateRequest(t, "/foo/bar", "GET", nil)
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
	client := NewOrDie(server.URL, nil)

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
	c.Validate(t, &response, err)
}
