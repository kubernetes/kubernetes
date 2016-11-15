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

package apiserver

import (
	"fmt"
	"net/http"
	"strings"

	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/storage"
	"k8s.io/kubernetes/pkg/util/runtime"
)

// statusError is an object that can be converted into an unversioned.Status
type statusError interface {
	Status() unversioned.Status
}

// errToAPIStatus converts an error to an unversioned.Status object.
func errToAPIStatus(err error) *unversioned.Status {
	switch t := err.(type) {
	case statusError:
		status := t.Status()
		if len(status.Status) == 0 {
			status.Status = unversioned.StatusFailure
		}
		if status.Code == 0 {
			switch status.Status {
			case unversioned.StatusSuccess:
				status.Code = http.StatusOK
			case unversioned.StatusFailure:
				status.Code = http.StatusInternalServerError
			}
		}
		//TODO: check for invalid responses
		return &status
	default:
		status := http.StatusInternalServerError
		switch {
		//TODO: replace me with NewConflictErr
		case storage.IsTestFailed(err):
			status = http.StatusConflict
		}
		// Log errors that were not converted to an error status
		// by REST storage - these typically indicate programmer
		// error by not using pkg/api/errors, or unexpected failure
		// cases.
		runtime.HandleError(fmt.Errorf("apiserver received an error that is not an unversioned.Status: %v", err))
		return &unversioned.Status{
			Status:  unversioned.StatusFailure,
			Code:    int32(status),
			Reason:  unversioned.StatusReasonUnknown,
			Message: err.Error(),
		}
	}
}

// notFound renders a simple not found error.
func notFound(w http.ResponseWriter, req *http.Request) {
	w.WriteHeader(http.StatusNotFound)
	fmt.Fprintf(w, "Not Found: %#v", req.RequestURI)
}

// internalError renders a simple internal error
func internalError(w http.ResponseWriter, req *http.Request, err error) {
	w.WriteHeader(http.StatusInternalServerError)
	fmt.Fprintf(w, "Internal Server Error: %#v", req.RequestURI)
	runtime.HandleError(err)
}

// errAPIPrefixNotFound indicates that a RequestInfo resolution failed because the request isn't under
// any known API prefixes
type errAPIPrefixNotFound struct {
	SpecifiedPrefix string
}

func (e *errAPIPrefixNotFound) Error() string {
	return fmt.Sprintf("no valid API prefix found matching %v", e.SpecifiedPrefix)
}

func IsAPIPrefixNotFound(err error) bool {
	if err == nil {
		return false
	}

	_, ok := err.(*errAPIPrefixNotFound)
	return ok
}

// errNotAcceptable indicates Accept negotiation has failed
// TODO: move to api/errors if other code needs to return this
type errNotAcceptable struct {
	accepted []string
}

func (e errNotAcceptable) Error() string {
	return fmt.Sprintf("only the following media types are accepted: %v", strings.Join(e.accepted, ", "))
}

func (e errNotAcceptable) Status() unversioned.Status {
	return unversioned.Status{
		Status:  unversioned.StatusFailure,
		Code:    http.StatusNotAcceptable,
		Reason:  unversioned.StatusReason("NotAcceptable"),
		Message: e.Error(),
	}
}

// errUnsupportedMediaType indicates Content-Type is not recognized
// TODO: move to api/errors if other code needs to return this
type errUnsupportedMediaType struct {
	accepted []string
}

func (e errUnsupportedMediaType) Error() string {
	return fmt.Sprintf("the body of the request was in an unknown format - accepted media types include: %v", strings.Join(e.accepted, ", "))
}

func (e errUnsupportedMediaType) Status() unversioned.Status {
	return unversioned.Status{
		Status:  unversioned.StatusFailure,
		Code:    http.StatusUnsupportedMediaType,
		Reason:  unversioned.StatusReason("UnsupportedMediaType"),
		Message: e.Error(),
	}
}
