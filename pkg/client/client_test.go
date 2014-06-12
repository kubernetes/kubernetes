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
	"net/http/httptest"
	"net/url"
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

// TODO: This doesn't reduce typing enough to make it worth the less readable errors. Remove.
func expectNoError(t *testing.T, err error) {
	if err != nil {
		t.Errorf("Unexpected error: %#v", err)
	}
}

// TODO: Move this to a common place, it's needed in multiple tests.
var apiPath = "/api/v1beta1"

func makeUrl(suffix string) string {
	return apiPath + suffix
}

func TestListEmptyPods(t *testing.T) {
	fakeHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: `{ "items": []}`,
	}
	testServer := httptest.NewTLSServer(&fakeHandler)
	client := Client{
		Host: testServer.URL,
	}
	podList, err := client.ListPods(nil)
	fakeHandler.ValidateRequest(t, makeUrl("/pods"), "GET", nil)
	if err != nil {
		t.Errorf("Unexpected error in listing pods: %#v", err)
	}
	if len(podList.Items) != 0 {
		t.Errorf("Unexpected items in pod list: %#v", podList)
	}
	testServer.Close()
}

func TestListPods(t *testing.T) {
	expectedPodList := api.PodList{
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
	}
	body, _ := json.Marshal(expectedPodList)
	fakeHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(body),
	}
	testServer := httptest.NewTLSServer(&fakeHandler)
	client := Client{
		Host: testServer.URL,
	}
	receivedPodList, err := client.ListPods(nil)
	fakeHandler.ValidateRequest(t, makeUrl("/pods"), "GET", nil)
	if err != nil {
		t.Errorf("Unexpected error in listing pods: %#v", err)
	}
	if !reflect.DeepEqual(expectedPodList, receivedPodList) {
		t.Errorf("Unexpected pod list: %#v\nvs.\n%#v", receivedPodList, expectedPodList)
	}
	testServer.Close()
}

func TestListPodsLabels(t *testing.T) {
	expectedPodList := api.PodList{
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
	}
	body, _ := json.Marshal(expectedPodList)
	fakeHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(body),
	}
	testServer := httptest.NewTLSServer(&fakeHandler)
	client := Client{
		Host: testServer.URL,
	}
	query := map[string]string{"foo": "bar", "name": "baz"}
	receivedPodList, err := client.ListPods(query)
	fakeHandler.ValidateRequest(t, makeUrl("/pods"), "GET", nil)
	queryString := fakeHandler.RequestReceived.URL.Query().Get("labels")
	queryString, _ = url.QueryUnescape(queryString)
	// TODO(bburns) : This assumes some ordering in serialization that might not always
	// be true, parse it into a map.
	if queryString != "foo=bar,name=baz" {
		t.Errorf("Unexpected label query: %s", queryString)
	}
	if err != nil {
		t.Errorf("Unexpected error in listing pods: %#v", err)
	}
	if !reflect.DeepEqual(expectedPodList, receivedPodList) {
		t.Errorf("Unexpected pod list: %#v\nvs.\n%#v", receivedPodList, expectedPodList)
	}
	testServer.Close()
}

func TestGetPod(t *testing.T) {
	expectedPod := api.Pod{
		CurrentState: api.PodState{
			Status: "Foobar",
		},
		Labels: map[string]string{
			"foo":  "bar",
			"name": "baz",
		},
	}
	body, _ := json.Marshal(expectedPod)
	fakeHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(body),
	}
	testServer := httptest.NewTLSServer(&fakeHandler)
	client := Client{
		Host: testServer.URL,
	}
	receivedPod, err := client.GetPod("foo")
	fakeHandler.ValidateRequest(t, makeUrl("/pods/foo"), "GET", nil)
	if err != nil {
		t.Errorf("Unexpected error: %#v", err)
	}
	if !reflect.DeepEqual(expectedPod, receivedPod) {
		t.Errorf("Received pod: %#v\n doesn't match expected pod: %#v", receivedPod, expectedPod)
	}
	testServer.Close()
}

func TestDeletePod(t *testing.T) {
	fakeHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: `{"success": true}`,
	}
	testServer := httptest.NewTLSServer(&fakeHandler)
	client := Client{
		Host: testServer.URL,
	}
	err := client.DeletePod("foo")
	fakeHandler.ValidateRequest(t, makeUrl("/pods/foo"), "DELETE", nil)
	if err != nil {
		t.Errorf("Unexpected error: %#v", err)
	}
	testServer.Close()
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
	body, _ := json.Marshal(requestPod)
	fakeHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(body),
	}
	testServer := httptest.NewTLSServer(&fakeHandler)
	client := Client{
		Host: testServer.URL,
	}
	receivedPod, err := client.CreatePod(requestPod)
	fakeHandler.ValidateRequest(t, makeUrl("/pods"), "POST", nil)
	if err != nil {
		t.Errorf("Unexpected error: %#v", err)
	}
	if !reflect.DeepEqual(requestPod, receivedPod) {
		t.Errorf("Received pod: %#v\n doesn't match expected pod: %#v", receivedPod, requestPod)
	}
	testServer.Close()
}

func TestUpdatePod(t *testing.T) {
	requestPod := api.Pod{
		JSONBase: api.JSONBase{ID: "foo"},
		CurrentState: api.PodState{
			Status: "Foobar",
		},
		Labels: map[string]string{
			"foo":  "bar",
			"name": "baz",
		},
	}
	body, _ := json.Marshal(requestPod)
	fakeHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(body),
	}
	testServer := httptest.NewTLSServer(&fakeHandler)
	client := Client{
		Host: testServer.URL,
	}
	receivedPod, err := client.UpdatePod(requestPod)
	fakeHandler.ValidateRequest(t, makeUrl("/pods/foo"), "PUT", nil)
	if err != nil {
		t.Errorf("Unexpected error: %#v", err)
	}
	expectEqual(t, requestPod, receivedPod)
	testServer.Close()
}

func expectEqual(t *testing.T, expected, observed interface{}) {
	if !reflect.DeepEqual(expected, observed) {
		t.Errorf("Unexpected inequality.  Expected: %#v Observed: %#v", expected, observed)
	}
}

func TestEncodeDecodeLabelQuery(t *testing.T) {
	queryIn := map[string]string{
		"foo": "bar",
		"baz": "blah",
	}
	queryString, _ := url.QueryUnescape(EncodeLabelQuery(queryIn))
	queryOut := DecodeLabelQuery(queryString)
	expectEqual(t, queryIn, queryOut)
}

func TestDecodeEmpty(t *testing.T) {
	query := DecodeLabelQuery("")
	if len(query) != 0 {
		t.Errorf("Unexpected query: %#v", query)
	}
}

func TestDecodeBad(t *testing.T) {
	query := DecodeLabelQuery("foo")
	if len(query) != 0 {
		t.Errorf("Unexpected query: %#v", query)
	}
}

func TestGetController(t *testing.T) {
	expectedController := api.ReplicationController{
		JSONBase: api.JSONBase{
			ID: "foo",
		},
		DesiredState: api.ReplicationControllerState{
			Replicas: 2,
		},
		Labels: map[string]string{
			"foo":  "bar",
			"name": "baz",
		},
	}
	body, _ := json.Marshal(expectedController)
	fakeHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(body),
	}
	testServer := httptest.NewTLSServer(&fakeHandler)
	client := Client{
		Host: testServer.URL,
	}
	receivedController, err := client.GetReplicationController("foo")
	expectNoError(t, err)
	if !reflect.DeepEqual(expectedController, receivedController) {
		t.Errorf("Unexpected controller, expected: %#v, received %#v", expectedController, receivedController)
	}
	fakeHandler.ValidateRequest(t, makeUrl("/replicationControllers/foo"), "GET", nil)
	testServer.Close()
}

func TestUpdateController(t *testing.T) {
	expectedController := api.ReplicationController{
		JSONBase: api.JSONBase{
			ID: "foo",
		},
		DesiredState: api.ReplicationControllerState{
			Replicas: 2,
		},
		Labels: map[string]string{
			"foo":  "bar",
			"name": "baz",
		},
	}
	body, _ := json.Marshal(expectedController)
	fakeHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(body),
	}
	testServer := httptest.NewTLSServer(&fakeHandler)
	client := Client{
		Host: testServer.URL,
	}
	receivedController, err := client.UpdateReplicationController(api.ReplicationController{
		JSONBase: api.JSONBase{
			ID: "foo",
		},
	})
	expectNoError(t, err)
	if !reflect.DeepEqual(expectedController, receivedController) {
		t.Errorf("Unexpected controller, expected: %#v, received %#v", expectedController, receivedController)
	}
	fakeHandler.ValidateRequest(t, makeUrl("/replicationControllers/foo"), "PUT", nil)
	testServer.Close()
}

func TestDeleteController(t *testing.T) {
	fakeHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: `{"success": true}`,
	}
	testServer := httptest.NewTLSServer(&fakeHandler)
	client := Client{
		Host: testServer.URL,
	}
	err := client.DeleteReplicationController("foo")
	fakeHandler.ValidateRequest(t, makeUrl("/replicationControllers/foo"), "DELETE", nil)
	if err != nil {
		t.Errorf("Unexpected error: %#v", err)
	}
	testServer.Close()
}

func TestCreateController(t *testing.T) {
	expectedController := api.ReplicationController{
		JSONBase: api.JSONBase{
			ID: "foo",
		},
		DesiredState: api.ReplicationControllerState{
			Replicas: 2,
		},
		Labels: map[string]string{
			"foo":  "bar",
			"name": "baz",
		},
	}
	body, _ := json.Marshal(expectedController)
	fakeHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(body),
	}
	testServer := httptest.NewTLSServer(&fakeHandler)
	client := Client{
		Host: testServer.URL,
	}
	receivedController, err := client.CreateReplicationController(api.ReplicationController{
		JSONBase: api.JSONBase{
			ID: "foo",
		},
	})
	expectNoError(t, err)
	if !reflect.DeepEqual(expectedController, receivedController) {
		t.Errorf("Unexpected controller, expected: %#v, received %#v", expectedController, receivedController)
	}
	fakeHandler.ValidateRequest(t, makeUrl("/replicationControllers"), "POST", nil)
	testServer.Close()
}
