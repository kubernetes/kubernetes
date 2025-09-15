/*
Copyright 2014 The Kubernetes Authors.

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

package responsewriters

import (
	stderrs "errors"
	"net/http"
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

func TestBadStatusErrorToAPIStatus(t *testing.T) {
	err := errors.StatusError{}
	actual := ErrorToAPIStatus(&err)
	expected := &metav1.Status{
		TypeMeta: metav1.TypeMeta{Kind: "Status", APIVersion: "v1"},
		Status:   metav1.StatusFailure,
		Code:     500,
	}
	if !reflect.DeepEqual(actual, expected) {
		t.Errorf("%s: Expected %#v, Got %#v", actual, expected, actual)
	}
}

func TestAPIStatus(t *testing.T) {
	cases := map[error]metav1.Status{
		errors.NewNotFound(schema.GroupResource{Group: "legacy.kubernetes.io", Resource: "foos"}, "bar"): {
			Status:  metav1.StatusFailure,
			Code:    http.StatusNotFound,
			Reason:  metav1.StatusReasonNotFound,
			Message: "foos.legacy.kubernetes.io \"bar\" not found",
			Details: &metav1.StatusDetails{
				Group: "legacy.kubernetes.io",
				Kind:  "foos",
				Name:  "bar",
			},
		},
		errors.NewAlreadyExists(schema.GroupResource{Resource: "foos"}, "bar"): {
			Status:  metav1.StatusFailure,
			Code:    http.StatusConflict,
			Reason:  "AlreadyExists",
			Message: "foos \"bar\" already exists",
			Details: &metav1.StatusDetails{
				Group: "",
				Kind:  "foos",
				Name:  "bar",
			},
		},
		errors.NewConflict(schema.GroupResource{Resource: "foos"}, "bar", stderrs.New("failure")): {
			Status:  metav1.StatusFailure,
			Code:    http.StatusConflict,
			Reason:  "Conflict",
			Message: "Operation cannot be fulfilled on foos \"bar\": failure",
			Details: &metav1.StatusDetails{
				Group: "",
				Kind:  "foos",
				Name:  "bar",
			},
		},
	}
	for k, v := range cases {
		actual := ErrorToAPIStatus(k)
		v.APIVersion = "v1"
		v.Kind = "Status"
		if !reflect.DeepEqual(actual, &v) {
			t.Errorf("%s: Expected %#v, Got %#v", k, v, actual)
		}
	}
}
