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
)

// errNotFound is an error which indicates that a specified resource is not found.
type errNotFound string

// Error returns a string representation of the err.
func (err errNotFound) Error() string {
	return string(err)
}

// IsNotFound determines if the err is an error which indicates that a specified resource was not found.
func IsNotFound(err error) bool {
	_, ok := err.(errNotFound)
	return ok
}

// NewNotFoundErr returns a new error which indicates that the resource of the kind and the name was not found.
func NewNotFoundErr(kind, name string) error {
	return errNotFound(fmt.Sprintf("%s %q not found", kind, name))
}

// errAlreadyExists is an error which indicates that a specified resource already exists.
type errAlreadyExists string

// Error returns a string representation of the err.
func (err errAlreadyExists) Error() string {
	return string(err)
}

// IsAlreadyExists determines if the err is an error which indicates that a specified resource already exists.
func IsAlreadyExists(err error) bool {
	_, ok := err.(errAlreadyExists)
	return ok
}

// NewAlreadyExistsErr returns a new error which indicates that the resource of the kind and the name was not found.
func NewAlreadyExistsErr(kind, name string) error {
	return errAlreadyExists(fmt.Sprintf("%s %q already exists", kind, name))
}

// internalError renders a generic error to the response
func internalError(err error, w http.ResponseWriter) {
	w.WriteHeader(http.StatusInternalServerError)
	fmt.Fprintf(w, "Internal Error: %#v", err)
}

// notFound renders a simple not found error
func notFound(w http.ResponseWriter, req *http.Request) {
	w.WriteHeader(http.StatusNotFound)
	fmt.Fprintf(w, "Not Found: %#v", req.RequestURI)
}
