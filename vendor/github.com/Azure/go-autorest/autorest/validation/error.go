package validation

// Copyright 2017 Microsoft Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

import (
	"fmt"
)

// Error is the type that's returned when the validation of an APIs arguments constraints fails.
type Error struct {
	// PackageType is the package type of the object emitting the error. For types, the value
	// matches that produced the the '%T' format specifier of the fmt package. For other elements,
	// such as functions, it is just the package name (e.g., "autorest").
	PackageType string

	// Method is the name of the method raising the error.
	Method string

	// Message is the error message.
	Message string
}

// Error returns a string containing the details of the validation failure.
func (e Error) Error() string {
	return fmt.Sprintf("%s#%s: Invalid input: %s", e.PackageType, e.Method, e.Message)
}

// NewError creates a new Error object with the specified parameters.
// message is treated as a format string to which the optional args apply.
func NewError(packageType string, method string, message string, args ...interface{}) Error {
	return Error{
		PackageType: packageType,
		Method:      method,
		Message:     fmt.Sprintf(message, args...),
	}
}
