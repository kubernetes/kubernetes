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
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/storage"
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

func TestConvertStorageErrorOrNot(t *testing.T) {
	cases := []struct {
		err            error
		expectedStatus metav1.Status
	}{
		{
			err: storage.NewKeyNotFoundError("", 0),
			expectedStatus: metav1.Status{
				Status:  metav1.StatusFailure,
				Code:    http.StatusNotFound,
				Reason:  "NotFound",
				Message: "StorageError: key not found, Code: 1, Key: , ResourceVersion: 0, AdditionalErrorMsg: ",
			},
		},
		{
			err: storage.NewKeyExistsError("", 0),
			expectedStatus: metav1.Status{
				Status:  metav1.StatusFailure,
				Code:    http.StatusConflict,
				Reason:  "AlreadyExists",
				Message: "StorageError: key exists, Code: 2, Key: , ResourceVersion: 0, AdditionalErrorMsg: ",
			},
		},
		{
			err: storage.NewUnreachableError("", 0),
			expectedStatus: metav1.Status{
				Status:  metav1.StatusFailure,
				Code:    http.StatusServiceUnavailable,
				Reason:  "ServiceUnavailable",
				Message: "StorageError: server unreachable, Code: 5, Key: , ResourceVersion: 0, AdditionalErrorMsg: ",
			},
		},
		{
			err: storage.NewResourceVersionConflictsError("", 0),
			expectedStatus: metav1.Status{
				Status:  metav1.StatusFailure,
				Code:    http.StatusConflict,
				Reason:  "Conflict",
				Message: "StorageError: resource version conflicts, Code: 3, Key: , ResourceVersion: 0, AdditionalErrorMsg: ",
			},
		},
		{
			err: storage.NewInvalidError([]*field.Error{field.Invalid(field.NewPath("test"), nil, "")}),
			expectedStatus: metav1.Status{
				Status:  metav1.StatusFailure,
				Code:    http.StatusUnprocessableEntity,
				Reason:  "Invalid",
				Message: "test: Invalid value: \"null\"",
			},
		},
	}
	for _, v := range cases {
		actual := convertStorageErrorOrNot(v.err)
		status := v.expectedStatus
		status.APIVersion = "v1"
		status.Kind = "Status"
		if !reflect.DeepEqual(actual, &status) {
			t.Errorf("%s: Expected %#v, Got %#v", v.err, v.expectedStatus, actual)
		}
	}
}
