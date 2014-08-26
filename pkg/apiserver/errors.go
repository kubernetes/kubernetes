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

package apiserver

import (
	"fmt"
	"net/http"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
)

// apiServerError is an error intended for consumption by a REST API server
type apiServerError struct {
	api.Status
}

// Error implements the Error interface.
func (e *apiServerError) Error() string {
	return e.Status.Message
}

// NewNotFoundErr returns a new error which indicates that the resource of the kind and the name was not found.
func NewNotFoundErr(kind, name string) error {
	return &apiServerError{api.Status{
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

// NewAlreadyExistsErr returns an error indicating the item requested exists by that identifier
func NewAlreadyExistsErr(kind, name string) error {
	return &apiServerError{api.Status{
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

// NewConflictErr returns an error indicating the item can't be updated as provided.
func NewConflictErr(kind, name string, err error) error {
	return &apiServerError{api.Status{
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

// NewInvalidError returns an error indicating the item is invalid and cannot be processed.
func NewInvalidErr(kind, name string, errs errors.ErrorList) error {
	causes := make([]api.StatusCause, 0, len(errs))
	for i := range errs {
		if err, ok := errs[i].(errors.ValidationError); ok {
			causes = append(causes, api.StatusCause{
				Type:    api.CauseType(err.Type),
				Message: err.Error(),
				Field:   err.Field,
			})
		}
	}
	return &apiServerError{api.Status{
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

// IsNotFound returns true if the specified error was created by NewNotFoundErr
func IsNotFound(err error) bool {
	return reasonForError(err) == api.StatusReasonNotFound
}

// IsAlreadyExists determines if the err is an error which indicates that a specified resource already exists.
func IsAlreadyExists(err error) bool {
	return reasonForError(err) == api.StatusReasonAlreadyExists
}

// IsConflict determines if the err is an error which indicates the provided update conflicts
func IsConflict(err error) bool {
	return reasonForError(err) == api.StatusReasonConflict
}

// IsInvalid determines if the err is an error which indicates the provided resource is not valid
func IsInvalid(err error) bool {
	return reasonForError(err) == api.StatusReasonInvalid
}

func reasonForError(err error) api.StatusReason {
	switch t := err.(type) {
	case *apiServerError:
		return t.Status.Reason
	}
	return api.StatusReasonUnknown
}

// errToAPIStatus converts an error to an api.Status object.
func errToAPIStatus(err error) *api.Status {
	switch t := err.(type) {
	case *apiServerError:
		status := t.Status
		status.Status = api.StatusFailure
		//TODO: check for invalid responses
		return &status
	default:
		status := http.StatusInternalServerError
		switch {
		//TODO: replace me with NewConflictErr
		case tools.IsEtcdTestFailed(err):
			status = http.StatusConflict
		}
		return &api.Status{
			Status:  api.StatusFailure,
			Code:    status,
			Reason:  api.StatusReasonUnknown,
			Message: err.Error(),
		}
	}
}

// notFound renders a simple not found error
func notFound(w http.ResponseWriter, req *http.Request) {
	w.WriteHeader(http.StatusNotFound)
	fmt.Fprintf(w, "Not Found: %#v", req.RequestURI)
}

// badGatewayError renders a simple bad gateway error
func badGatewayError(w http.ResponseWriter, req *http.Request) {
	w.WriteHeader(http.StatusBadGateway)
	fmt.Fprintf(w, "Bad Gateway: %#v", req.RequestURI)
}
