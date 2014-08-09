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
		Reason: api.ReasonTypeNotFound,
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
		Reason: api.ReasonTypeAlreadyExists,
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
		Reason: api.ReasonTypeConflict,
		Details: &api.StatusDetails{
			Kind: kind,
			ID:   name,
		},
		Message: fmt.Sprintf("%s %q cannot be updated: %s", kind, name, err),
	}}
}

// IsNotFound returns true if the specified error was created by NewNotFoundErr
func IsNotFound(err error) bool {
	return reasonForError(err) == api.ReasonTypeNotFound
}

// IsAlreadyExists determines if the err is an error which indicates that a specified resource already exists.
func IsAlreadyExists(err error) bool {
	return reasonForError(err) == api.ReasonTypeAlreadyExists
}

// IsConflict determines if the err is an error which indicates the provided update conflicts
func IsConflict(err error) bool {
	return reasonForError(err) == api.ReasonTypeConflict
}

func reasonForError(err error) api.ReasonType {
	switch t := err.(type) {
	case *apiServerError:
		return t.Status.Reason
	}
	return api.ReasonTypeUnknown
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
		//TODO: replace me with NewUpdateConflictErr
		case tools.IsEtcdTestFailed(err):
			status = http.StatusConflict
		}
		return &api.Status{
			Status:  api.StatusFailure,
			Code:    status,
			Reason:  api.ReasonTypeUnknown,
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
