/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/fielderrors"
)

func TestErrorNew(t *testing.T) {
	err := NewAlreadyExists("test", "1")
	if !IsAlreadyExists(err) {
		t.Errorf("expected to be %s", api.StatusReasonAlreadyExists)
	}
	if IsConflict(err) {
		t.Errorf("expected to not be %s", api.StatusReasonConflict)
	}
	if IsNotFound(err) {
		t.Errorf(fmt.Sprintf("expected to not be %s", api.StatusReasonNotFound))
	}
	if IsInvalid(err) {
		t.Errorf("expected to not be %s", api.StatusReasonInvalid)
	}
	if IsBadRequest(err) {
		t.Errorf("expected to not be %s", api.StatusReasonBadRequest)
	}
	if IsForbidden(err) {
		t.Errorf("expected to not be %s", api.StatusReasonForbidden)
	}
	if IsServerTimeout(err) {
		t.Errorf("expected to not be %s", api.StatusReasonServerTimeout)
	}
	if IsMethodNotSupported(err) {
		t.Errorf("expected to not be %s", api.StatusReasonMethodNotAllowed)
	}

	if !IsConflict(NewConflict("test", "2", errors.New("message"))) {
		t.Errorf("expected to be conflict")
	}
	if !IsNotFound(NewNotFound("test", "3")) {
		t.Errorf("expected to be %s", api.StatusReasonNotFound)
	}
	if !IsInvalid(NewInvalid("test", "2", nil)) {
		t.Errorf("expected to be %s", api.StatusReasonInvalid)
	}
	if !IsBadRequest(NewBadRequest("reason")) {
		t.Errorf("expected to be %s", api.StatusReasonBadRequest)
	}
	if !IsForbidden(NewForbidden("test", "2", errors.New("reason"))) {
		t.Errorf("expected to be %s", api.StatusReasonForbidden)
	}
	if !IsUnauthorized(NewUnauthorized("reason")) {
		t.Errorf("expected to be %s", api.StatusReasonUnauthorized)
	}
	if !IsServerTimeout(NewServerTimeout("test", "reason", 0)) {
		t.Errorf("expected to be %s", api.StatusReasonServerTimeout)
	}
	if time, ok := SuggestsClientDelay(NewServerTimeout("test", "doing something", 10)); time != 10 || !ok {
		t.Errorf("expected to be %s", api.StatusReasonServerTimeout)
	}
	if time, ok := SuggestsClientDelay(NewTimeoutError("test reason", 10)); time != 10 || !ok {
		t.Errorf("expected to be %s", api.StatusReasonTimeout)
	}
	if !IsMethodNotSupported(NewMethodNotSupported("foo", "delete")) {
		t.Errorf("expected to be %s", api.StatusReasonMethodNotAllowed)
	}
}

func TestNewInvalid(t *testing.T) {
	testCases := []struct {
		Err     *fielderrors.ValidationError
		Details *api.StatusDetails
	}{
		{
			fielderrors.NewFieldDuplicate("field[0].name", "bar"),
			&api.StatusDetails{
				Kind: "kind",
				Name: "name",
				Causes: []api.StatusCause{{
					Type:  api.CauseTypeFieldValueDuplicate,
					Field: "field[0].name",
				}},
			},
		},
		{
			fielderrors.NewFieldInvalid("field[0].name", "bar", "detail"),
			&api.StatusDetails{
				Kind: "kind",
				Name: "name",
				Causes: []api.StatusCause{{
					Type:  api.CauseTypeFieldValueInvalid,
					Field: "field[0].name",
				}},
			},
		},
		{
			fielderrors.NewFieldNotFound("field[0].name", "bar"),
			&api.StatusDetails{
				Kind: "kind",
				Name: "name",
				Causes: []api.StatusCause{{
					Type:  api.CauseTypeFieldValueNotFound,
					Field: "field[0].name",
				}},
			},
		},
		{
			fielderrors.NewFieldValueNotSupported("field[0].name", "bar", nil),
			&api.StatusDetails{
				Kind: "kind",
				Name: "name",
				Causes: []api.StatusCause{{
					Type:  api.CauseTypeFieldValueNotSupported,
					Field: "field[0].name",
				}},
			},
		},
		{
			fielderrors.NewFieldRequired("field[0].name"),
			&api.StatusDetails{
				Kind: "kind",
				Name: "name",
				Causes: []api.StatusCause{{
					Type:  api.CauseTypeFieldValueRequired,
					Field: "field[0].name",
				}},
			},
		},
	}
	for i, testCase := range testCases {
		vErr, expected := testCase.Err, testCase.Details
		expected.Causes[0].Message = vErr.ErrorBody()
		err := NewInvalid("kind", "name", fielderrors.ValidationErrorList{vErr})
		status := err.(*StatusError).ErrStatus
		if status.Code != 422 || status.Reason != api.StatusReasonInvalid {
			t.Errorf("%d: unexpected status: %#v", i, status)
		}
		if !reflect.DeepEqual(expected, status.Details) {
			t.Errorf("%d: expected %#v, got %#v", i, expected, status.Details)
		}
	}
}

func Test_reasonForError(t *testing.T) {
	if e, a := api.StatusReasonUnknown, reasonForError(nil); e != a {
		t.Errorf("unexpected reason type: %#v", a)
	}
}

type TestType struct{}

func (*TestType) IsAnAPIObject() {}

func TestFromObject(t *testing.T) {
	table := []struct {
		obj     runtime.Object
		message string
	}{
		{&api.Status{Message: "foobar"}, "foobar"},
		{&TestType{}, "unexpected object: &{}"},
	}

	for _, item := range table {
		if e, a := item.message, FromObject(item.obj).Error(); e != a {
			t.Errorf("Expected %v, got %v", e, a)
		}
	}
}
