package restful

// Copyright 2013 Ernest Micklei. All rights reserved.
// Use of this source code is governed by a license
// that can be found in the LICENSE file.

import (
	"net/http"
)

// DefaultContainer is a restful.Container that uses http.DefaultServeMux
var DefaultContainer *Container

func init() {
	DefaultContainer = NewContainer()
	DefaultContainer.ServeMux = http.DefaultServeMux
}

// If set the true then panics will not be caught to return HTTP 500.
// In that case, Route functions are responsible for handling any error situation.
// Default value is false = recover from panics. This has performance implications.
// OBSOLETE ; use restful.DefaultContainer.DoNotRecover(true)
var DoNotRecover = false

// Add registers a new WebService add it to the DefaultContainer.
func Add(service *WebService) {
	DefaultContainer.Add(service)
}

// Filter appends a container FilterFunction from the DefaultContainer.
// These are called before dispatching a http.Request to a WebService.
func Filter(filter FilterFunction) {
	DefaultContainer.Filter(filter)
}

// RegisteredWebServices returns the collections of WebServices from the DefaultContainer
func RegisteredWebServices() []*WebService {
	return DefaultContainer.RegisteredWebServices()
}
