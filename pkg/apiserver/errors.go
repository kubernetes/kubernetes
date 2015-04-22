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

package apiserver

import (
	"fmt"
	"net/http"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

// statusError is an object that can be converted into an api.Status
type statusError interface {
	Status() api.Status
}

// errToAPIStatus converts an error to an api.Status object.
func errToAPIStatus(err error) *api.Status {
	switch t := err.(type) {
	case statusError:
		status := t.Status()
		if len(status.Status) == 0 {
			status.Status = api.StatusFailure
		}
		if status.Code == 0 {
			switch status.Status {
			case api.StatusSuccess:
				status.Code = http.StatusOK
			case api.StatusFailure:
				status.Code = http.StatusInternalServerError
			}
		}
		//TODO: check for invalid responses
		return &status
	default:
		status := http.StatusInternalServerError
		switch {
		//TODO: replace me with NewConflictErr
		case tools.IsEtcdTestFailed(err):
			status = http.StatusConflict
		}
		// Log errors that were not converted to an error status
		// by REST storage - these typically indicate programmer
		// error by not using pkg/api/errors, or unexpected failure
		// cases.
		util.HandleError(fmt.Errorf("apiserver received an error that is not an api.Status: %v", err))
		return &api.Status{
			Status:  api.StatusFailure,
			Code:    status,
			Reason:  api.StatusReasonUnknown,
			Message: err.Error(),
		}
	}
}

// notFound renders a simple not found error.
func notFound(w http.ResponseWriter, req *http.Request) {
	w.WriteHeader(http.StatusNotFound)
	fmt.Fprintf(w, "Not Found: %#v", req.RequestURI)
}

// badGatewayError renders a simple bad gateway error.
func badGatewayError(w http.ResponseWriter, req *http.Request) {
	w.WriteHeader(http.StatusBadGateway)
	fmt.Fprintf(w, "Bad Gateway: %#v", req.RequestURI)
}

// forbidden renders a simple forbidden error
func forbidden(w http.ResponseWriter, req *http.Request) {
	w.WriteHeader(http.StatusForbidden)
	fmt.Fprintf(w, "Forbidden: %#v", req.RequestURI)
}
