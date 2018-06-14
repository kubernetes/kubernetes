/*
Copyright 2018 The Kubernetes Authors.

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

package cloud

import "fmt"

// OperationPollingError occurs when the GCE Operation cannot be retrieved for a prolonged period.
type OperationPollingError struct {
	LastPollError error
}

// Error returns a string representation including the last poll error encountered.
func (e *OperationPollingError) Error() string {
	return fmt.Sprintf("GCE operation polling error: %v", e.LastPollError)
}

// GCEOperationError occurs when the GCE Operation finishes with an error.
type GCEOperationError struct {
	// HTTPStatusCode is the HTTP status code of the final error.
	// For example, a failed operation may have 400 - BadRequest.
	HTTPStatusCode int
	// Code is GCE's code of what went wrong.
	// For example, RESOURCE_IN_USE_BY_ANOTHER_RESOURCE
	Code string
	// Message is a human readable message.
	// For example, "The network resource 'xxx' is already being used by 'xxx'"
	Message string
}

// Error returns a string representation including the HTTP Status code, GCE's error code
// and a human readable message.
func (e *GCEOperationError) Error() string {
	return fmt.Sprintf("GCE %v - %v: %v", e.HTTPStatusCode, e.Code, e.Message)
}
