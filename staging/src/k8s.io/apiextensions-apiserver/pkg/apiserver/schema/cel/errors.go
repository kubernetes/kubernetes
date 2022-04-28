/*
Copyright 2021 The Kubernetes Authors.

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

package cel

// Error is an implementation of the 'error' interface, which represents a
// XValidation error.
type Error struct {
	Type   ErrorType
	Detail string
}

var _ error = &Error{}

// Error implements the error interface.
func (v *Error) Error() string {
	return v.Detail
}

// ErrorType is a machine readable value providing more detail about why
// a XValidation is invalid.
type ErrorType string

const (
	// ErrorTypeRequired is used to report withNullable values that are not
	// provided (e.g. empty strings, null values, or empty arrays).  See
	// Required().
	ErrorTypeRequired ErrorType = "RuleRequired"
	// ErrorTypeInvalid is used to report malformed values
	ErrorTypeInvalid ErrorType = "RuleInvalid"
	// ErrorTypeInternal is used to report other errors that are not related
	// to user input.  See InternalError().
	ErrorTypeInternal ErrorType = "InternalError"
)
