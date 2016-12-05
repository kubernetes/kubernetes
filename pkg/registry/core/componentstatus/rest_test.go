/*
Copyright 2015 The Kubernetes Authors.

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
	"fmt"
	"reflect"
	"strings"
	"testing"

	"net/http"
	"net/url"
	"time"

	"k8s.io/kubernetes/pkg/api"
	metav1 "k8s.io/kubernetes/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/apiserver"
	"k8s.io/kubernetes/pkg/probe"
	"k8s.io/kubernetes/pkg/util/diff"
)

type fakeHttpProber struct {
	result probe.Result
	body   string
	err    error
}

func (f *fakeHttpProber) Probe(*url.URL, http.Header, time.Duration) (probe.Result, string, error) {
	return f.result, f.body, f.err
}

type testResponse struct {
	result probe.Result
	data   string
	err    error
}

func NewTestREST(resp testResponse) *REST {
	return &REST{
		GetServersToValidate: func() map[string]apiserver.Server {
			return map[string]apiserver.Server{
				"test1": {Addr: "testserver1", Port: 8000, Path: "/healthz"},
			}
		},
		prober: &fakeHttpProber{
			result: resp.result,
			body:   resp.data,
			err:    resp.err,
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
	r := NewTestREST(testResponse{result: probe.Success, data: "ok"})
	got, err := r.List(api.NewContext(), nil)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	expect := &api.ComponentStatusList{
		Items: []api.ComponentStatus{*(createTestStatus("test1", api.ConditionTrue, "ok", ""))},
	}
	if e, a := expect, got; !reflect.DeepEqual(e, a) {
		t.Errorf("Got unexpected object. Diff: %s", diff.ObjectDiff(e, a))
	}
}

func TestList_FailedCheck(t *testing.T) {
	r := NewTestREST(testResponse{result: probe.Failure, data: ""})
	got, err := r.List(api.NewContext(), nil)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	expect := &api.ComponentStatusList{
		Items: []api.ComponentStatus{
			*(createTestStatus("test1", api.ConditionFalse, "", ""))},
	}
	if e, a := expect, got; !reflect.DeepEqual(e, a) {
		t.Errorf("Got unexpected object. Diff: %s", diff.ObjectDiff(e, a))
	}
}

func TestList_UnknownError(t *testing.T) {
	r := NewTestREST(testResponse{result: probe.Unknown, data: "", err: fmt.Errorf("fizzbuzz error")})
	got, err := r.List(api.NewContext(), nil)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	expect := &api.ComponentStatusList{
		Items: []api.ComponentStatus{
			*(createTestStatus("test1", api.ConditionUnknown, "", "fizzbuzz error"))},
	}
	if e, a := expect, got; !reflect.DeepEqual(e, a) {
		t.Errorf("Got unexpected object. Diff: %s", diff.ObjectDiff(e, a))
	}
}

func TestGet_NoError(t *testing.T) {
	r := NewTestREST(testResponse{result: probe.Success, data: "ok"})
	got, err := r.Get(api.NewContext(), "test1", &metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	expect := createTestStatus("test1", api.ConditionTrue, "ok", "")
	if e, a := expect, got; !reflect.DeepEqual(e, a) {
		t.Errorf("Got unexpected object. Diff: %s", diff.ObjectDiff(e, a))
	}
}

func TestGet_BadName(t *testing.T) {
	r := NewTestREST(testResponse{result: probe.Success, data: "ok"})
	_, err := r.Get(api.NewContext(), "invalidname", &metav1.GetOptions{})
	if err == nil {
		t.Fatalf("Expected error, but did not get one")
	}
	if !strings.Contains(err.Error(), "Component not found: invalidname") {
		t.Fatalf("Got unexpected error: %v", err)
	}
}
