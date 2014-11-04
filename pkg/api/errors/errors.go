/*
Copyright 2014 Google Inc. All rights reserved.

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
	"fmt"
	"net/http"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
)

// statusError is an error intended for consumption by a REST API server.
type statusError struct {
	status api.Status
}

// Error implements the Error interface.
func (e *statusError) Error() string {
	return e.status.Message
}

// Status converts this error into an api.Status object.
func (e *statusError) Status() api.Status {
	return e.status
}

// FromObject generates an statusError from an api.Status, if that is the type of obj; otherwise,
// returns an error created by fmt.Errorf.
func FromObject(obj runtime.Object) error {
	switch t := obj.(type) {
	case *api.Status:
		return &statusError{*t}
	}
	return fmt.Errorf("unexpected object: %v", obj)
}

// NewNotFound returns a new error which indicates that the resource of the kind and the name was not found.
func NewNotFound(kind, name string) error {
	return &statusError{api.Status{
		Status: api.StatusFailure,
		Code:   http.StatusNotFound,
		Reason: api.StatusReasonNotFound,
		Details: &api.StatusDetails{
			Kind: kind,
			ID:   name,
		},
		Message: fmt.Sprintf("%s %q not found", kind, name),
	}}
}

// NewAlreadyExists returns an error indicating the item requested exists by that identifier.
func NewAlreadyExists(kind, name string) error {
	return &statusError{api.Status{
		Status: api.StatusFailure,
		Code:   http.StatusConflict,
		Reason: api.StatusReasonAlreadyExists,
		Details: &api.StatusDetails{
			Kind: kind,
			ID:   name,
		},
		Message: fmt.Sprintf("%s %q already exists", kind, name),
	}}
}

// NewConflict returns an error indicating the item can't be updated as provided.
func NewConflict(kind, name string, err error) error {
	return &statusError{api.Status{
		Status: api.StatusFailure,
		Code:   http.StatusConflict,
		Reason: api.StatusReasonConflict,
		Details: &api.StatusDetails{
			Kind: kind,
			ID:   name,
		},
		Message: fmt.Sprintf("%s %q cannot be updated: %s", kind, name, err),
	}}
}

// NewInvalid returns an error indicating the item is invalid and cannot be processed.
func NewInvalid(kind, name string, errs ValidationErrorList) error {
	causes := make([]api.StatusCause, 0, len(errs))
	for i := range errs {
		if err, ok := errs[i].(ValidationError); ok {
			causes = append(causes, api.StatusCause{
				Type:    api.CauseType(err.Type),
				Message: err.Error(),
				Field:   err.Field,
			})
		}
	}
	return &statusError{api.Status{
		Status: api.StatusFailure,
		Code:   422, // RFC 4918
		Reason: api.StatusReasonInvalid,
		Details: &api.StatusDetails{
			Kind:   kind,
			ID:     name,
			Causes: causes,
		},
		Message: fmt.Sprintf("%s %q is invalid: %s", kind, name, errs.ToError()),
	}}
}

// NewBadRequest creates an error that indicates that the request is invalid and can not be processed.
func NewBadRequest(reason string) error {
	return &statusError{
		api.Status{
			Status: api.StatusFailure,
			Code:   400,
			Reason: api.StatusReasonBadRequest,
			Details: &api.StatusDetails{
				Causes: []api.StatusCause{
					{Message: reason},
				},
			},
		},
	}
}

// NewInternalError returns an error indicating the item is invalid and cannot be processed.
func NewInternalError(err error) error {
	return &statusError{api.Status{
		Status: api.StatusFailure,
		Code:   500,
		Reason: api.StatusReasonInternalError,
		Details: &api.StatusDetails{
			Causes: []api.StatusCause{{Message: err.Error()}},
		},
		Message: fmt.Sprintf("Internal error occurred: %v", err),
	}}
}

// IsNotFound returns true if the specified error was created by NewNotFoundErr.
func IsNotFound(err error) bool {
	return reasonForError(err) == api.StatusReasonNotFound
}

// IsAlreadyExists determines if the err is an error which indicates that a specified resource already exists.
func IsAlreadyExists(err error) bool {
	return reasonForError(err) == api.StatusReasonAlreadyExists
}

// IsConflict determines if the err is an error which indicates the provided update conflicts.
func IsConflict(err error) bool {
	return reasonForError(err) == api.StatusReasonConflict
}

// IsInvalid determines if the err is an error which indicates the provided resource is not valid.
func IsInvalid(err error) bool {
	return reasonForError(err) == api.StatusReasonInvalid
}

// IsBadRequest determines if err is an error which indicates that the request is invalid.
func IsBadRequest(err error) bool {
	return reasonForError(err) == api.StatusReasonBadRequest
}

func reasonForError(err error) api.StatusReason {
	switch t := err.(type) {
	case *statusError:
		return t.status.Reason
	}
	return api.StatusReasonUnknown
}
