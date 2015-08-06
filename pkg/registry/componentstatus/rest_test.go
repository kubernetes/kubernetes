/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package componentstatus

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"net/http"
	"reflect"
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apiserver"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util"
)

type fakeRoundTripper struct {
	err  error
	resp *http.Response
	url  string
}

func (f *fakeRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	f.url = req.URL.String()
	return f.resp, f.err
}

type testResponse struct {
	code int
	data string
	err  error
}

func NewTestREST(resp testResponse) *REST {
	return &REST{
		GetServersToValidate: func() map[string]apiserver.Server {
			return map[string]apiserver.Server{
				"test1": {Addr: "testserver1", Port: 8000, Path: "/healthz"},
			}
		},
		rt: &fakeRoundTripper{
			err: resp.err,
			resp: &http.Response{
				Body:       ioutil.NopCloser(bytes.NewBufferString(resp.data)),
				StatusCode: resp.code,
			},
		},
	}
}

func createTestStatus(name string, status api.ConditionStatus, msg string, err string) *api.ComponentStatus {
	retVal := &api.ComponentStatus{
		Conditions: []api.ComponentCondition{
			{Type: api.ComponentHealthy, Status: status, Message: msg, Error: err},
		},
	}
	retVal.Name = name
	return retVal
}

func TestList_NoError(t *testing.T) {
	r := NewTestREST(testResponse{code: 200, data: "ok"})
	got, err := r.List(api.NewContext(), labels.Everything(), fields.Everything())
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	expect := &api.ComponentStatusList{
		Items: []api.ComponentStatus{*(createTestStatus("test1", api.ConditionTrue, "ok", "nil"))},
	}
	if e, a := expect, got; !reflect.DeepEqual(e, a) {
		t.Errorf("Got unexpected object. Diff: %s", util.ObjectDiff(e, a))
	}
}

func TestList_FailedCheck(t *testing.T) {
	r := NewTestREST(testResponse{code: 500, data: ""})
	got, err := r.List(api.NewContext(), labels.Everything(), fields.Everything())
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	expect := &api.ComponentStatusList{
		Items: []api.ComponentStatus{
			*(createTestStatus("test1", api.ConditionFalse, "", "unhealthy http status code: 500 ()"))},
	}
	if e, a := expect, got; !reflect.DeepEqual(e, a) {
		t.Errorf("Got unexpected object. Diff: %s", util.ObjectDiff(e, a))
	}
}

func TestList_UnknownError(t *testing.T) {
	r := NewTestREST(testResponse{code: 500, data: "", err: fmt.Errorf("fizzbuzz error")})
	got, err := r.List(api.NewContext(), labels.Everything(), fields.Everything())
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	expect := &api.ComponentStatusList{
		Items: []api.ComponentStatus{
			*(createTestStatus("test1", api.ConditionUnknown, "", "Get http://testserver1:8000/healthz: fizzbuzz error"))},
	}
	if e, a := expect, got; !reflect.DeepEqual(e, a) {
		t.Errorf("Got unexpected object. Diff: %s", util.ObjectDiff(e, a))
	}
}

func TestGet_NoError(t *testing.T) {
	r := NewTestREST(testResponse{code: 200, data: "ok"})
	got, err := r.Get(api.NewContext(), "test1")
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	expect := createTestStatus("test1", api.ConditionTrue, "ok", "nil")
	if e, a := expect, got; !reflect.DeepEqual(e, a) {
		t.Errorf("Got unexpected object. Diff: %s", util.ObjectDiff(e, a))
	}
}

func TestGet_BadName(t *testing.T) {
	r := NewTestREST(testResponse{code: 200, data: "ok"})
	_, err := r.Get(api.NewContext(), "invalidname")
	if err == nil {
		t.Fatalf("Expected error, but did not get one")
	}
	if !strings.Contains(err.Error(), "Component not found: invalidname") {
		t.Fatalf("Got unexpected error: %v", err)
	}
}
