/*
Copyright The Kubernetes Authors.

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

package remotecommand

import (
	"fmt"
	"net/http"
)

const (
	statusSuccess = "Success"
	statusFailure = "Failure"

	statusReasonInternalError = "InternalError"
)

// streamStatusError carries status details written to the error stream.
type streamStatusError struct {
	ErrStatus streamStatus
}

func (e *streamStatusError) Error() string {
	return e.ErrStatus.Message
}

func (e *streamStatusError) status() streamStatus {
	return e.ErrStatus
}

func newInternalError(err error) *streamStatusError {
	return &streamStatusError{
		ErrStatus: streamStatus{
			Status:  statusFailure,
			Reason:  statusReasonInternalError,
			Message: fmt.Sprintf("Internal error occurred: %v", err),
			Code:    http.StatusInternalServerError,
		},
	}
}

type streamStatus struct {
	Status  string               `json:"status,omitempty"`
	Message string               `json:"message,omitempty"`
	Reason  string               `json:"reason,omitempty"`
	Details *streamStatusDetails `json:"details,omitempty"`
	Code    int32                `json:"code,omitempty"`
}

type streamStatusDetails struct {
	Causes []streamStatusCause `json:"causes,omitempty"`
}

// streamStatusCause uses "reason" on the wire to match metav1.StatusCause.
type streamStatusCause struct {
	Type    string `json:"reason,omitempty"`
	Message string `json:"message,omitempty"`
}
