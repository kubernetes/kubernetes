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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/validation/field"
)

func TestErrorNew(t *testing.T) {
	err := NewAlreadyExists(api.Resource("tests"), "1")
	if !IsAlreadyExists(err) {
		t.Errorf("expected to be %s", unversioned.StatusReasonAlreadyExists)
	}
	if IsConflict(err) {
		t.Errorf("expected to not be %s", unversioned.StatusReasonConflict)
	}
	if IsNotFound(err) {
		t.Errorf(fmt.Sprintf("expected to not be %s", unversioned.StatusReasonNotFound))
	}
	if IsInvalid(err) {
		t.Errorf("expected to not be %s", unversioned.StatusReasonInvalid)
	}
	if IsBadRequest(err) {
		t.Errorf("expected to not be %s", unversioned.StatusReasonBadRequest)
	}
	if IsForbidden(err) {
		t.Errorf("expected to not be %s", unversioned.StatusReasonForbidden)
	}
	if IsServerTimeout(err) {
		t.Errorf("expected to not be %s", unversioned.StatusReasonServerTimeout)
	}
	if IsMethodNotSupported(err) {
		t.Errorf("expected to not be %s", unversioned.StatusReasonMethodNotAllowed)
	}

	if !IsConflict(NewConflict(api.Resource("tests"), "2", errors.New("message"))) {
		t.Errorf("expected to be conflict")
	}
	if !IsNotFound(NewNotFound(api.Resource("tests"), "3")) {
		t.Errorf("expected to be %s", unversioned.StatusReasonNotFound)
	}
	if !IsInvalid(NewInvalid(api.Kind("Test"), "2", nil)) {
		t.Errorf("expected to be %s", unversioned.StatusReasonInvalid)
	}
	if !IsBadRequest(NewBadRequest("reason")) {
		t.Errorf("expected to be %s", unversioned.StatusReasonBadRequest)
	}
	if !IsForbidden(NewForbidden(api.Resource("tests"), "2", errors.New("reason"))) {
		t.Errorf("expected to be %s", unversioned.StatusReasonForbidden)
	}
	if !IsUnauthorized(NewUnauthorized("reason")) {
		t.Errorf("expected to be %s", unversioned.StatusReasonUnauthorized)
	}
	if !IsServerTimeout(NewServerTimeout(api.Resource("tests"), "reason", 0)) {
		t.Errorf("expected to be %s", unversioned.StatusReasonServerTimeout)
	}
	if time, ok := SuggestsClientDelay(NewServerTimeout(api.Resource("tests"), "doing something", 10)); time != 10 || !ok {
		t.Errorf("expected to be %s", unversioned.StatusReasonServerTimeout)
	}
	if time, ok := SuggestsClientDelay(NewTimeoutError("test reason", 10)); time != 10 || !ok {
		t.Errorf("expected to be %s", unversioned.StatusReasonTimeout)
	}
	if !IsMethodNotSupported(NewMethodNotSupported(api.Resource("foos"), "delete")) {
		t.Errorf("expected to be %s", unversioned.StatusReasonMethodNotAllowed)
	}
}

func TestNewInvalid(t *testing.T) {
	testCases := []struct {
		Err     *field.Error
		Details *unversioned.StatusDetails
	}{
		{
			field.Duplicate(field.NewPath("field[0].name"), "bar"),
			&unversioned.StatusDetails{
				Kind: "Kind",
				Name: "name",
				Causes: []unversioned.StatusCause{{
					Type:  unversioned.CauseTypeFieldValueDuplicate,
					Field: "field[0].name",
				}},
			},
		},
		{
			field.Invalid(field.NewPath("field[0].name"), "bar", "detail"),
			&unversioned.StatusDetails{
				Kind: "Kind",
				Name: "name",
				Causes: []unversioned.StatusCause{{
					Type:  unversioned.CauseTypeFieldValueInvalid,
					Field: "field[0].name",
				}},
			},
		},
		{
			field.NotFound(field.NewPath("field[0].name"), "bar"),
			&unversioned.StatusDetails{
				Kind: "Kind",
				Name: "name",
				Causes: []unversioned.StatusCause{{
					Type:  unversioned.CauseTypeFieldValueNotFound,
					Field: "field[0].name",
				}},
			},
		},
		{
			field.NotSupported(field.NewPath("field[0].name"), "bar", nil),
			&unversioned.StatusDetails{
				Kind: "Kind",
				Name: "name",
				Causes: []unversioned.StatusCause{{
					Type:  unversioned.CauseTypeFieldValueNotSupported,
					Field: "field[0].name",
				}},
			},
		},
		{
			field.Required(field.NewPath("field[0].name"), ""),
			&unversioned.StatusDetails{
				Kind: "Kind",
				Name: "name",
				Causes: []unversioned.StatusCause{{
					Type:  unversioned.CauseTypeFieldValueRequired,
					Field: "field[0].name",
				}},
			},
		},
	}
	for i, testCase := range testCases {
		vErr, expected := testCase.Err, testCase.Details
		expected.Causes[0].Message = vErr.ErrorBody()
		err := NewInvalid(api.Kind("Kind"), "name", field.ErrorList{vErr})
		status := err.ErrStatus
		if status.Code != 422 || status.Reason != unversioned.StatusReasonInvalid {
			t.Errorf("%d: unexpected status: %#v", i, status)
		}
		if !reflect.DeepEqual(expected, status.Details) {
			t.Errorf("%d: expected %#v, got %#v", i, expected, status.Details)
		}
	}
}

func Test_reasonForError(t *testing.T) {
	if e, a := unversioned.StatusReasonUnknown, reasonForError(nil); e != a {
		t.Errorf("unexpected reason type: %#v", a)
	}
}

type TestType struct{}

func (obj *TestType) GetObjectKind() unversioned.ObjectKind { return unversioned.EmptyObjectKind }

func TestFromObject(t *testing.T) {
	table := []struct {
		obj     runtime.Object
		message string
	}{
		{&unversioned.Status{Message: "foobar"}, "foobar"},
		{&TestType{}, "unexpected object: &{}"},
	}

	for _, item := range table {
		if e, a := item.message, FromObject(item.obj).Error(); e != a {
			t.Errorf("Expected %v, got %v", e, a)
		}
	}
}
