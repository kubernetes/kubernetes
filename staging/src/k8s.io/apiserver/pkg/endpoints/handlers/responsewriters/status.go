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
	"fmt"
	"net/http"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apiserver/pkg/storage"
)

// statusError is an object that can be converted into an metav1.Status
type statusError interface {
	Status() metav1.Status
}

// ErrorToAPIStatus converts an error to an metav1.Status object.
func ErrorToAPIStatus(err error) *metav1.Status {
	switch t := err.(type) {
	case statusError:
		status := t.Status()
		if len(status.Status) == 0 {
			status.Status = metav1.StatusFailure
		}
		switch status.Status {
		case metav1.StatusSuccess:
			if status.Code == 0 {
				status.Code = http.StatusOK
			}
		case metav1.StatusFailure:
			if status.Code == 0 {
				status.Code = http.StatusInternalServerError
			}
		default:
			runtime.HandleError(fmt.Errorf("apiserver received an error with wrong status field : %#+v", err))
			if status.Code == 0 {
				status.Code = http.StatusInternalServerError
			}
		}
		status.Kind = "Status"
		status.APIVersion = "v1"
		//TODO: check for invalid responses
		return &status
	default:
		status := http.StatusInternalServerError
		switch {
		//TODO: replace me with NewConflictErr
		case storage.IsConflict(err):
			status = http.StatusConflict
		}
		// Log errors that were not converted to an error status
		// by REST storage - these typically indicate programmer
		// error by not using pkg/api/errors, or unexpected failure
		// cases.
		runtime.HandleError(fmt.Errorf("apiserver received an error that is not an metav1.Status: %#+v", err))
		return &metav1.Status{
			TypeMeta: metav1.TypeMeta{
				Kind:       "Status",
				APIVersion: "v1",
			},
			Status:  metav1.StatusFailure,
			Code:    int32(status),
			Reason:  metav1.StatusReasonUnknown,
			Message: err.Error(),
		}
	}
}
