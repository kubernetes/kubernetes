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

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"

	"net/http"
	"net/url"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	metav1beta1 "k8s.io/apimachinery/pkg/apis/meta/v1beta1"
	"k8s.io/apimachinery/pkg/util/diff"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/probe"
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
	prober := &fakeHttpProber{
		result: resp.result,
		body:   resp.data,
		err:    resp.err,
	}
	return NewStorage(func() map[string]*Server {
			return map[string]*Server{
				"test1": {Addr: "testserver1", Port: 8000, Path: "/healthz", Prober: prober},
			}
	})
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
	got, err := r.List(genericapirequest.NewContext(), nil)
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

func TestList_WithLabelSelectors(t *testing.T) {
	r := NewTestREST(testResponse{result: probe.Success, data: "ok"})
	opts := metainternalversion.ListOptions{
		LabelSelector: labels.SelectorFromSet(map[string]string{
			"testLabel": "testValue",
		}),
	}
	got, err := r.List(genericapirequest.NewContext(), &opts)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	expect := &api.ComponentStatusList{
		Items: []api.ComponentStatus{},
	}
	if e, a := expect, got; !reflect.DeepEqual(e, a) {
		t.Errorf("Got unexpected object. Diff: %s", diff.ObjectDiff(e, a))
	}
}

func TestList_WithFieldSelectors(t *testing.T) {
	r := NewTestREST(testResponse{result: probe.Success, data: "ok"})
	opts := metainternalversion.ListOptions{
		FieldSelector: fields.SelectorFromSet(map[string]string{
			"testField": "testValue",
		}),
	}
	got, err := r.List(genericapirequest.NewContext(), &opts)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	expect := &api.ComponentStatusList{
		Items: []api.ComponentStatus{},
	}
	if e, a := expect, got; !reflect.DeepEqual(e, a) {
		t.Errorf("Got unexpected object. Diff: %s", diff.ObjectDiff(e, a))
	}
}

func TestList_FailedCheck(t *testing.T) {
	r := NewTestREST(testResponse{result: probe.Failure, data: ""})
	got, err := r.List(genericapirequest.NewContext(), nil)
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
	got, err := r.List(genericapirequest.NewContext(), nil)
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
	got, err := r.Get(genericapirequest.NewContext(), "test1", &metav1.GetOptions{})
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
	_, err := r.Get(genericapirequest.NewContext(), "invalidname", &metav1.GetOptions{})
	if err == nil {
		t.Fatalf("Expected error, but did not get one")
	}
	if !strings.Contains(err.Error(), "Component not found: invalidname") {
		t.Fatalf("Got unexpected error: %v", err)
	}
}

func TestConvertToTable(t *testing.T) {
	r := NewTestREST(testResponse{result: probe.Success, data: "ok"})
	columns := []metav1beta1.TableColumnDefinition{
		{Name: "Name", Type: "string", Format: "name", Description: metav1.ObjectMeta{}.SwaggerDoc()["name"]},
		{Name: "Status", Type: "string", Description: "Status of the component conditions"},
		{Name: "Message", Type: "string", Description: "Message of the component conditions"},
		{Name: "Error", Type: "string", Description: "Error of the component conditions"},
	}

	componentStatus1 := createTestStatus("test1", api.ConditionUnknown, "", "fizzbuzz error")
	componentStatus2 := createTestStatus("test2", api.ConditionTrue, "ok", "")

	testCases := []struct {
		in  runtime.Object
		out *metav1beta1.Table
		err bool
	}{
		{
			in:  nil,
			err: true,
		},
		{
			in: &api.ComponentStatusList{},
			out: &metav1beta1.Table{ColumnDefinitions: columns},
		},
		{
			in: &api.ComponentStatusList{
				Items: []api.ComponentStatus{
					*componentStatus1,
					*componentStatus2,
				},
			},
			out: &metav1beta1.Table{
				ColumnDefinitions: columns,
				Rows: []metav1beta1.TableRow{
					{Cells: []interface{}{"test1", "Unhealthy", "", "fizzbuzz error"}, Object: runtime.RawExtension{Object: componentStatus1}},
					{Cells: []interface{}{"test2", "Healthy", "ok", ""}, Object: runtime.RawExtension{Object: componentStatus2}},
				},
			},
		},
	}

	ctx := genericapirequest.NewDefaultContext()
	for i, test := range testCases {
		out, err := r.ConvertToTable(ctx, test.in, nil)
		if err != nil {
			if test.err {
				continue
			}
			t.Errorf("%d: error: %v", i, err)
			continue
		}
		if !apiequality.Semantic.DeepEqual(test.out, out) {
			t.Errorf("%d: mismatch: %s", i, diff.ObjectReflectDiff(test.out, out))
		}
	}
}