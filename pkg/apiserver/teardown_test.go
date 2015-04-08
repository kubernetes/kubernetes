/*
Copyright 2015 Google Inc. All rights reserved.

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

package apiserver

import (
	"errors"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
)

type deleteSpec struct {
	name      string
	namespace string
	retval    error
}

type testServiceREST struct {
	services       api.ServiceList
	deletes        []deleteSpec
	nextDelete     int
	expectedResult []error
	t              *testing.T
}

func (t *testServiceREST) NewList() runtime.Object {
	return nil
}

func (t *testServiceREST) List(ctx api.Context, label, field labels.Selector) (runtime.Object, error) {
	return &t.services, nil
}

func (t *testServiceREST) Delete(ctx api.Context, name string) (runtime.Object, error) {
	if t.nextDelete >= len(t.deletes) {
		t.t.Errorf("Trying to delete more services than expected")
		return nil, nil
	}
	namespace, _ := api.NamespaceFrom(ctx)
	if t.deletes[t.nextDelete].namespace != namespace {
		t.t.Errorf("Unexpected namespace during delete, got: %v expected: %v", namespace, t.deletes[t.nextDelete].namespace)
	}
	if t.deletes[t.nextDelete].name != name {
		t.t.Errorf("Unexpected name during delete, got: %v expected: %v", name, t.deletes[t.nextDelete].name)
	}
	t.nextDelete++
	return nil, t.deletes[t.nextDelete-1].retval
}

func (t *testServiceREST) verifyResult(result []error) {
	if t.nextDelete != len(t.deletes) {
		t.t.Errorf("Some services were not deleted, deleted: %v, expected: %v", t.nextDelete, len(t.deletes))
	}
	if len(t.expectedResult) != len(result) {
		t.t.Errorf("Results differ, got: %v expected: %v", result, t.expectedResult)
	} else {
		for i, r := range t.expectedResult {
			if r.Error() != result[i].Error() {
				t.t.Errorf("Results differ at index %v, got: %v expected: %v", i, result[i], r)
			}
		}
	}
}

func TestTeardown(t *testing.T) {
	tests := []testServiceREST{
		// One service w/external load balancer, should be deleted.
		{
			services: api.ServiceList{
				Items: []api.Service{{
					ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: "default"},
					Spec:       api.ServiceSpec{CreateExternalLoadBalancer: true}}},
			},
			deletes:        []deleteSpec{{"abc", "default", nil}},
			nextDelete:     0,
			expectedResult: []error{},
			t:              t,
		},
		// One service w/o external load balancer, should not be deleted.
		{
			services: api.ServiceList{
				Items: []api.Service{{
					ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: "default"},
					Spec:       api.ServiceSpec{CreateExternalLoadBalancer: false}}},
			},
			deletes:        []deleteSpec{},
			nextDelete:     0,
			expectedResult: []error{},
			t:              t,
		},
		// Error during deletion.
		{
			services: api.ServiceList{
				Items: []api.Service{{
					ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: "default"},
					Spec:       api.ServiceSpec{CreateExternalLoadBalancer: true}}},
			},
			deletes:        []deleteSpec{{"abc", "default", errors.New("error")}},
			nextDelete:     0,
			expectedResult: []error{errors.New("error")},
			t:              t,
		},
		// Three services in different namespaces, two w/external load balancer, one returns error during deletion.
		{
			services: api.ServiceList{
				Items: []api.Service{
					{
						ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: "n1"},
						Spec:       api.ServiceSpec{CreateExternalLoadBalancer: true},
					},
					{
						ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: "n2"},
						Spec:       api.ServiceSpec{CreateExternalLoadBalancer: false},
					},
					{
						ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: "n3"},
						Spec:       api.ServiceSpec{CreateExternalLoadBalancer: true},
					},
				},
			},
			deletes: []deleteSpec{
				{"abc", "n1", nil},
				{"abc", "n3", errors.New("error-n3")},
			},
			nextDelete:     0,
			expectedResult: []error{errors.New("error-n3")},
			t:              t,
		},
	}
	for _, test := range tests {
		h, _ := NewTeardownHandler(&test)
		th, ok := h.(*teardownHandler)
		if !ok || th == nil {
			t.Error("NewTeardownHandler() returned type other than *teardownHandler")
			return
		}
		res := th.teardown()
		test.verifyResult(res)
	}
}
