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

package errors

import (
	"errors"
	"fmt"
	"reflect"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

func resource(resource string) schema.GroupResource {
	return schema.GroupResource{Group: "", Resource: resource}
}
func kind(kind string) schema.GroupKind {
	return schema.GroupKind{Group: "", Kind: kind}
}

func TestErrorNew(t *testing.T) {
	err := NewAlreadyExists(resource("tests"), "1")
	if !IsAlreadyExists(err) {
		t.Errorf("expected to be %s", metav1.StatusReasonAlreadyExists)
	}
	if IsConflict(err) {
		t.Errorf("expected to not be %s", metav1.StatusReasonConflict)
	}
	if IsNotFound(err) {
		t.Errorf(fmt.Sprintf("expected to not be %s", metav1.StatusReasonNotFound))
	}
	if IsInvalid(err) {
		t.Errorf("expected to not be %s", metav1.StatusReasonInvalid)
	}
	if IsBadRequest(err) {
		t.Errorf("expected to not be %s", metav1.StatusReasonBadRequest)
	}
	if IsForbidden(err) {
		t.Errorf("expected to not be %s", metav1.StatusReasonForbidden)
	}
	if IsServerTimeout(err) {
		t.Errorf("expected to not be %s", metav1.StatusReasonServerTimeout)
	}
	if IsMethodNotSupported(err) {
		t.Errorf("expected to not be %s", metav1.StatusReasonMethodNotAllowed)
	}

	if !IsConflict(NewConflict(resource("tests"), "2", errors.New("message"))) {
		t.Errorf("expected to be conflict")
	}
	if !IsNotFound(NewNotFound(resource("tests"), "3")) {
		t.Errorf("expected to be %s", metav1.StatusReasonNotFound)
	}
	if !IsInvalid(NewInvalid(kind("Test"), "2", nil)) {
		t.Errorf("expected to be %s", metav1.StatusReasonInvalid)
	}
	if !IsBadRequest(NewBadRequest("reason")) {
		t.Errorf("expected to be %s", metav1.StatusReasonBadRequest)
	}
	if !IsForbidden(NewForbidden(resource("tests"), "2", errors.New("reason"))) {
		t.Errorf("expected to be %s", metav1.StatusReasonForbidden)
	}
	if !IsUnauthorized(NewUnauthorized("reason")) {
		t.Errorf("expected to be %s", metav1.StatusReasonUnauthorized)
	}
	if !IsServerTimeout(NewServerTimeout(resource("tests"), "reason", 0)) {
		t.Errorf("expected to be %s", metav1.StatusReasonServerTimeout)
	}
	if !IsMethodNotSupported(NewMethodNotSupported(resource("foos"), "delete")) {
		t.Errorf("expected to be %s", metav1.StatusReasonMethodNotAllowed)
	}

	if time, ok := SuggestsClientDelay(NewServerTimeout(resource("tests"), "doing something", 10)); time != 10 || !ok {
		t.Errorf("unexpected %d", time)
	}
	if time, ok := SuggestsClientDelay(NewServerTimeout(resource("tests"), "doing something", 0)); time != 0 || !ok {
		t.Errorf("unexpected %d", time)
	}
	if time, ok := SuggestsClientDelay(NewTimeoutError("test reason", 10)); time != 10 || !ok {
		t.Errorf("unexpected %d", time)
	}
	if time, ok := SuggestsClientDelay(NewTooManyRequests("doing something", 10)); time != 10 || !ok {
		t.Errorf("unexpected %d", time)
	}
	if time, ok := SuggestsClientDelay(NewTooManyRequests("doing something", 1)); time != 1 || !ok {
		t.Errorf("unexpected %d", time)
	}
	if time, ok := SuggestsClientDelay(NewGenericServerResponse(429, "get", resource("tests"), "test", "doing something", 10, true)); time != 10 || !ok {
		t.Errorf("unexpected %d", time)
	}
	if time, ok := SuggestsClientDelay(NewGenericServerResponse(500, "get", resource("tests"), "test", "doing something", 10, true)); time != 10 || !ok {
		t.Errorf("unexpected %d", time)
	}
	if time, ok := SuggestsClientDelay(NewGenericServerResponse(429, "get", resource("tests"), "test", "doing something", 0, true)); time != 0 || ok {
		t.Errorf("unexpected %d", time)
	}
}

func TestNewInvalid(t *testing.T) {
	testCases := []struct {
		Err     *field.Error
		Details *metav1.StatusDetails
	}{
		{
			field.Duplicate(field.NewPath("field[0].name"), "bar"),
			&metav1.StatusDetails{
				Kind: "Kind",
				Name: "name",
				Causes: []metav1.StatusCause{{
					Type:  metav1.CauseTypeFieldValueDuplicate,
					Field: "field[0].name",
				}},
			},
		},
		{
			field.Invalid(field.NewPath("field[0].name"), "bar", "detail"),
			&metav1.StatusDetails{
				Kind: "Kind",
				Name: "name",
				Causes: []metav1.StatusCause{{
					Type:  metav1.CauseTypeFieldValueInvalid,
					Field: "field[0].name",
				}},
			},
		},
		{
			field.NotFound(field.NewPath("field[0].name"), "bar"),
			&metav1.StatusDetails{
				Kind: "Kind",
				Name: "name",
				Causes: []metav1.StatusCause{{
					Type:  metav1.CauseTypeFieldValueNotFound,
					Field: "field[0].name",
				}},
			},
		},
		{
			field.NotSupported(field.NewPath("field[0].name"), "bar", nil),
			&metav1.StatusDetails{
				Kind: "Kind",
				Name: "name",
				Causes: []metav1.StatusCause{{
					Type:  metav1.CauseTypeFieldValueNotSupported,
					Field: "field[0].name",
				}},
			},
		},
		{
			field.Required(field.NewPath("field[0].name"), ""),
			&metav1.StatusDetails{
				Kind: "Kind",
				Name: "name",
				Causes: []metav1.StatusCause{{
					Type:  metav1.CauseTypeFieldValueRequired,
					Field: "field[0].name",
				}},
			},
		},
	}
	for i, testCase := range testCases {
		vErr, expected := testCase.Err, testCase.Details
		expected.Causes[0].Message = vErr.ErrorBody()
		err := NewInvalid(kind("Kind"), "name", field.ErrorList{vErr})
		status := err.ErrStatus
		if status.Code != 422 || status.Reason != metav1.StatusReasonInvalid {
			t.Errorf("%d: unexpected status: %#v", i, status)
		}
		if !reflect.DeepEqual(expected, status.Details) {
			t.Errorf("%d: expected %#v, got %#v", i, expected, status.Details)
		}
	}
}

func Test_reasonForError(t *testing.T) {
	if e, a := metav1.StatusReasonUnknown, reasonForError(nil); e != a {
		t.Errorf("unexpected reason type: %#v", a)
	}
}

type TestType struct{}

func (obj *TestType) GetObjectKind() schema.ObjectKind { return schema.EmptyObjectKind }
func (obj *TestType) DeepCopyObject() runtime.Object {
	if obj == nil {
		return nil
	}
	clone := *obj
	return &clone
}

func TestFromObject(t *testing.T) {
	table := []struct {
		obj     runtime.Object
		message string
	}{
		{&metav1.Status{Message: "foobar"}, "foobar"},
		{&TestType{}, "unexpected object: &{}"},
	}

	for _, item := range table {
		if e, a := item.message, FromObject(item.obj).Error(); e != a {
			t.Errorf("Expected %v, got %v", e, a)
		}
	}
}
