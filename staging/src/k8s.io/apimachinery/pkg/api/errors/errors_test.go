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
	"net/http"
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

func TestReasonForError(t *testing.T) {
	if e, a := metav1.StatusReasonUnknown, ReasonForError(nil); e != a {
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

func TestReasonForErrorSupportsWrappedErrors(t *testing.T) {
	testCases := []struct {
		name           string
		err            error
		expectedReason metav1.StatusReason
	}{
		{
			name:           "Direct match",
			err:            &StatusError{ErrStatus: metav1.Status{Reason: metav1.StatusReasonUnauthorized}},
			expectedReason: metav1.StatusReasonUnauthorized,
		},
		{
			name:           "No match",
			err:            errors.New("some other error"),
			expectedReason: metav1.StatusReasonUnknown,
		},
		{
			name:           "Nested match",
			err:            fmt.Errorf("wrapping: %w", fmt.Errorf("some more: %w", &StatusError{ErrStatus: metav1.Status{Reason: metav1.StatusReasonAlreadyExists}})),
			expectedReason: metav1.StatusReasonAlreadyExists,
		},
		{
			name:           "Nested, no match",
			err:            fmt.Errorf("wrapping: %w", fmt.Errorf("some more: %w", errors.New("hello"))),
			expectedReason: metav1.StatusReasonUnknown,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			if result := ReasonForError(tc.err); result != tc.expectedReason {
				t.Errorf("expected reason: %q, but got known reason: %q", tc.expectedReason, result)
			}
		})
	}
}

func TestIsTooManyRequestsSupportsWrappedErrors(t *testing.T) {
	testCases := []struct {
		name        string
		err         error
		expectMatch bool
	}{
		{
			name:        "Direct match via status reason",
			err:         &StatusError{ErrStatus: metav1.Status{Reason: metav1.StatusReasonTooManyRequests}},
			expectMatch: true,
		},
		{
			name:        "Direct match via status code",
			err:         &StatusError{ErrStatus: metav1.Status{Code: http.StatusTooManyRequests}},
			expectMatch: true,
		},
		{
			name:        "No match",
			err:         &StatusError{},
			expectMatch: false,
		},
		{
			name:        "Nested match via status reason",
			err:         fmt.Errorf("Wrapping: %w", &StatusError{ErrStatus: metav1.Status{Reason: metav1.StatusReasonTooManyRequests}}),
			expectMatch: true,
		},
		{
			name:        "Nested match via status code",
			err:         fmt.Errorf("Wrapping: %w", &StatusError{ErrStatus: metav1.Status{Code: http.StatusTooManyRequests}}),
			expectMatch: true,
		},
		{
			name:        "Nested,no match",
			err:         fmt.Errorf("Wrapping: %w", &StatusError{ErrStatus: metav1.Status{Code: http.StatusNotFound}}),
			expectMatch: false,
		},
	}

	for _, tc := range testCases {
		if result := IsTooManyRequests(tc.err); result != tc.expectMatch {
			t.Errorf("Expect match %t, got match %t", tc.expectMatch, result)
		}
	}
}
func TestIsRequestEntityTooLargeErrorSupportsWrappedErrors(t *testing.T) {
	testCases := []struct {
		name        string
		err         error
		expectMatch bool
	}{
		{
			name:        "Direct match via status reason",
			err:         &StatusError{ErrStatus: metav1.Status{Reason: metav1.StatusReasonRequestEntityTooLarge}},
			expectMatch: true,
		},
		{
			name:        "Direct match via status code",
			err:         &StatusError{ErrStatus: metav1.Status{Code: http.StatusRequestEntityTooLarge}},
			expectMatch: true,
		},
		{
			name:        "No match",
			err:         &StatusError{},
			expectMatch: false,
		},
		{
			name:        "Nested match via status reason",
			err:         fmt.Errorf("Wrapping: %w", &StatusError{ErrStatus: metav1.Status{Reason: metav1.StatusReasonRequestEntityTooLarge}}),
			expectMatch: true,
		},
		{
			name:        "Nested match via status code",
			err:         fmt.Errorf("Wrapping: %w", &StatusError{ErrStatus: metav1.Status{Code: http.StatusRequestEntityTooLarge}}),
			expectMatch: true,
		},
		{
			name:        "Nested,no match",
			err:         fmt.Errorf("Wrapping: %w", &StatusError{ErrStatus: metav1.Status{Code: http.StatusNotFound}}),
			expectMatch: false,
		},
	}

	for _, tc := range testCases {
		if result := IsRequestEntityTooLargeError(tc.err); result != tc.expectMatch {
			t.Errorf("Expect match %t, got match %t", tc.expectMatch, result)
		}
	}
}

func TestIsUnexpectedServerError(t *testing.T) {
	unexpectedServerErr := func() error {
		return &StatusError{
			ErrStatus: metav1.Status{
				Details: &metav1.StatusDetails{
					Causes: []metav1.StatusCause{{Type: metav1.CauseTypeUnexpectedServerResponse}},
				},
			},
		}
	}
	testCases := []struct {
		name        string
		err         error
		expectMatch bool
	}{
		{
			name:        "Direct match",
			err:         unexpectedServerErr(),
			expectMatch: true,
		},
		{
			name:        "No match",
			err:         errors.New("some other error"),
			expectMatch: false,
		},
		{
			name:        "Nested match",
			err:         fmt.Errorf("wrapping: %w", unexpectedServerErr()),
			expectMatch: true,
		},
		{
			name:        "Nested, no match",
			err:         fmt.Errorf("wrapping: %w", fmt.Errorf("some more: %w", errors.New("hello"))),
			expectMatch: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			if result := IsUnexpectedServerError(tc.err); result != tc.expectMatch {
				t.Errorf("expected match: %t, but got match: %t", tc.expectMatch, result)
			}
		})
	}
}

func TestIsUnexpectedObjectError(t *testing.T) {
	unexpectedObjectErr := func() error {
		return &UnexpectedObjectError{}
	}
	testCases := []struct {
		name        string
		err         error
		expectMatch bool
	}{
		{
			name:        "Direct match",
			err:         unexpectedObjectErr(),
			expectMatch: true,
		},
		{
			name:        "No match",
			err:         errors.New("some other error"),
			expectMatch: false,
		},
		{
			name:        "Nested match",
			err:         fmt.Errorf("wrapping: %w", unexpectedObjectErr()),
			expectMatch: true,
		},
		{
			name:        "Nested, no match",
			err:         fmt.Errorf("wrapping: %w", fmt.Errorf("some more: %w", errors.New("hello"))),
			expectMatch: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			if result := IsUnexpectedObjectError(tc.err); result != tc.expectMatch {
				t.Errorf("expected match: %t, but got match: %t", tc.expectMatch, result)
			}
		})
	}
}

func TestSuggestsClientDelaySupportsWrapping(t *testing.T) {
	suggestsClientDelayErr := func() error {
		return &StatusError{
			ErrStatus: metav1.Status{
				Reason:  metav1.StatusReasonServerTimeout,
				Details: &metav1.StatusDetails{},
			},
		}
	}
	testCases := []struct {
		name        string
		err         error
		expectMatch bool
	}{
		{
			name:        "Direct match",
			err:         suggestsClientDelayErr(),
			expectMatch: true,
		},
		{
			name:        "No match",
			err:         errors.New("some other error"),
			expectMatch: false,
		},
		{
			name:        "Nested match",
			err:         fmt.Errorf("wrapping: %w", suggestsClientDelayErr()),
			expectMatch: true,
		},
		{
			name:        "Nested, no match",
			err:         fmt.Errorf("wrapping: %w", fmt.Errorf("some more: %w", errors.New("hello"))),
			expectMatch: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			if _, result := SuggestsClientDelay(tc.err); result != tc.expectMatch {
				t.Errorf("expected match: %t, but got match: %t", tc.expectMatch, result)
			}
		})
	}
}
