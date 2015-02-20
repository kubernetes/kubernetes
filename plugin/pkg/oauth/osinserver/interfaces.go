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

package osinserver

import (
	"net/http"

	"github.com/RangelReale/osin"
)

// Mux is an interface describing the methods Install requires
type Mux interface {
	HandleFunc(pattern string, handler func(http.ResponseWriter, *http.Request))
}

// AuthorizeHandler populates an AuthorizeRequest or handles the request itself
type AuthorizeHandler interface {
	// HandleAuthorize populates an AuthorizeRequest (typically the Authorized and UserData fields)
	// and returns false, or handles the request itself and returns true.
	HandleAuthorize(ar *osin.AuthorizeRequest, w http.ResponseWriter) (handled bool, err error)
}

// AccessHandler populates an AccessRequest
type AccessHandler interface {
	// HandleAccess populates an AccessRequest (typically the Authorized and UserData fields)
	HandleAccess(ar *osin.AccessRequest, w http.ResponseWriter) error
}

// ErrorHandler writes an error response
type ErrorHandler interface {
	// HandleError writes an error response
	HandleError(err error, w http.ResponseWriter, req *http.Request)
}

// AuthorizeHandlerFunc adapts a function into an AuthorizeHandler
type AuthorizeHandlerFunc func(ar *osin.AuthorizeRequest, w http.ResponseWriter) (bool, error)

// HandleAuthorize implements osinserver.AuthorizeHandler
func (f AuthorizeHandlerFunc) HandleAuthorize(ar *osin.AuthorizeRequest, w http.ResponseWriter) (bool, error) {
	return f(ar, w)
}

// AuthorizeHandlers is a chain of AuthorizeHandler objects. The chain short-circuits
// if any handler handles the request.
type AuthorizeHandlers []AuthorizeHandler

// HandleAuthorize implements osinserver.AuthorizeHandler
func (all AuthorizeHandlers) HandleAuthorize(ar *osin.AuthorizeRequest, w http.ResponseWriter) (handled bool, err error) {
	for _, h := range all {
		handled, err := h.HandleAuthorize(ar, w)
		if err != nil || handled {
			return handled, err
		}
	}
	return false, nil
}

// AccessHandlerFunc adapts a function into an AccessHandler
type AccessHandlerFunc func(ar *osin.AccessRequest, w http.ResponseWriter) error

// HandleAccess implements osinserver.AccessHandler
func (f AccessHandlerFunc) HandleAccess(ar *osin.AccessRequest, w http.ResponseWriter) error {
	return f(ar, w)
}

// AccessHandlers is a chain of AccessHandler objects
type AccessHandlers []AccessHandler

// HandleAccess implements osinserver.AccessHandler
func (all AccessHandlers) HandleAccess(ar *osin.AccessRequest, w http.ResponseWriter) error {
	for _, h := range all {
		if err := h.HandleAccess(ar, w); err != nil {
			return err
		}
	}
	return nil
}
