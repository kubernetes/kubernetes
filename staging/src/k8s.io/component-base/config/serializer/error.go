/*
Copyright 2019 The Kubernetes Authors.

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

package serializer

import (
	"fmt"

	"k8s.io/apimachinery/pkg/runtime/schema"
)

// Error implements the error interface
var _ error = &UnrecognizedVersionError{}

// UnrecognizedVersionError is a base error type that is returned when decoding bytes that
// use a too old API version.
type UnrecognizedVersionError struct {
	message      string
	gvk          schema.GroupVersionKind
	originalData []byte
}

// NewUnrecognizedVersionError creates a new UnrecognizedVersionError object
func NewUnrecognizedVersionError(message string, gvk schema.GroupVersionKind, originalData []byte) *UnrecognizedVersionError {
	return &UnrecognizedVersionError{
		message:      message,
		gvk:          gvk,
		originalData: originalData,
	}
}

// Error implements the error interface
func (e *UnrecognizedVersionError) Error() string {
	return fmt.Sprintf("unrecognized version %s in known group %s for kind %v: %s", e.gvk.Version, e.gvk.Group, e.gvk, e.message)
}

// GVK returns the GroupVersionKind for this error
func (e *UnrecognizedVersionError) GVK() schema.GroupVersionKind {
	return e.gvk
}

// OriginalData returns the original byte slice input.
func (e *UnrecognizedVersionError) OriginalData() []byte {
	return e.originalData
}

// IsUnrecognizedVersionError returns true if the error...
func IsUnrecognizedVersionError(err error) bool {
	if err == nil {
		return false
	}
	_, ok := err.(*UnrecognizedVersionError)
	return ok
}
