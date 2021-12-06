package restful

// Copyright 2013 Ernest Micklei. All rights reserved.
// Use of this source code is governed by a license
// that can be found in the LICENSE file.

import (
	"net/http"
	"strings"
)

// RouteFunction declares the signature of a function that can be bound to a Route.
type RouteFunction func(*Request, *Response)

// RouteSelectionConditionFunction declares the signature of a function that
// can be used to add extra conditional logic when selecting whether the route
// matches the HTTP request.
type RouteSelectionConditionFunction func(httpRequest *http.Request) bool

// Route binds a HTTP Method,Path,Consumes combination to a RouteFunction.
type Route struct {
	ExtensionProperties
	Method   string
	Produces []string
	Consumes []string
	Path     string // webservice root path + described path
	Function RouteFunction
	Filters  []FilterFunction
	If       []RouteSelectionConditionFunction

	// cached values for dispatching
	relativePath string
	pathParts    []string
	pathExpr     *pathExpression // cached compilation of relativePath as RegExp

	// documentation
	Doc                     string
	Notes                   string
	Operation               string
	ParameterDocs           []*Parameter
	ResponseErrors          map[int]ResponseError
	DefaultResponse         *ResponseError
	ReadSample, WriteSample interface{} // structs that model an example request or response payload

	// Extra information used to store custom information about the route.
	Metadata map[string]interface{}

	// marks a route as deprecated
	Deprecated bool

	//Overrides the container.contentEncodingEnabled
	contentEncodingEnabled *bool

	// indicate route path has custom verb
	hasCustomVerb bool

	// if a request does not include a content-type header then
	// depending on the method, it may return a 415 Unsupported Media
	// Must have uppercase HTTP Method names such as GET,HEAD,OPTIONS,...
	allowedMethodsWithoutContentType []string
}

// Initialize for Route
func (r *Route) postBuild() {
	r.pathParts = tokenizePath(r.Path)
	r.hasCustomVerb = hasCustomVerb(r.Path)
}

// Create Request and Response from their http versions
func (r *Route) wrapRequestResponse(httpWriter http.ResponseWriter, httpRequest *http.Request, pathParams map[string]string) (*Request, *Response) {
	wrappedRequest := NewRequest(httpRequest)
	wrappedRequest.pathParameters = pathParams
	wrappedRequest.selectedRoute = r
	wrappedResponse := NewResponse(httpWriter)
	wrappedResponse.requestAccept = httpRequest.Header.Get(HEADER_Accept)
	wrappedResponse.routeProduces = r.Produces
	return wrappedRequest, wrappedResponse
}

func stringTrimSpaceCutset(r rune) bool {
	return r == ' '
}

// Return whether the mimeType matches to what this Route can produce.
func (r Route) matchesAccept(mimeTypesWithQuality string) bool {
	remaining := mimeTypesWithQuality
	for {
		var mimeType string
		if end := strings.Index(remaining, ","); end == -1 {
			mimeType, remaining = remaining, ""
		} else {
			mimeType, remaining = remaining[:end], remaining[end+1:]
		}
		if quality := strings.Index(mimeType, ";"); quality != -1 {
			mimeType = mimeType[:quality]
		}
		mimeType = strings.TrimFunc(mimeType, stringTrimSpaceCutset)
		if mimeType == "*/*" {
			return true
		}
		for _, producibleType := range r.Produces {
			if producibleType == "*/*" || producibleType == mimeType {
				return true
			}
		}
		if len(remaining) == 0 {
			return false
		}
	}
}

// Return whether this Route can consume content with a type specified by mimeTypes (can be empty).
func (r Route) matchesContentType(mimeTypes string) bool {

	if len(r.Consumes) == 0 {
		// did not specify what it can consume ;  any media type (“*/*”) is assumed
		return true
	}

	if len(mimeTypes) == 0 {
		// idempotent methods with (most-likely or guaranteed) empty content match missing Content-Type
		m := r.Method
		// if route specifies less or non-idempotent methods then use that
		if len(r.allowedMethodsWithoutContentType) > 0 {
			for _, each := range r.allowedMethodsWithoutContentType {
				if m == each {
					return true
				}
			}
		} else {
			if m == "GET" || m == "HEAD" || m == "OPTIONS" || m == "DELETE" || m == "TRACE" {
				return true
			}
		}
		// proceed with default
		mimeTypes = MIME_OCTET
	}

	remaining := mimeTypes
	for {
		var mimeType string
		if end := strings.Index(remaining, ","); end == -1 {
			mimeType, remaining = remaining, ""
		} else {
			mimeType, remaining = remaining[:end], remaining[end+1:]
		}
		if quality := strings.Index(mimeType, ";"); quality != -1 {
			mimeType = mimeType[:quality]
		}
		mimeType = strings.TrimFunc(mimeType, stringTrimSpaceCutset)
		for _, consumeableType := range r.Consumes {
			if consumeableType == "*/*" || consumeableType == mimeType {
				return true
			}
		}
		if len(remaining) == 0 {
			return false
		}
	}
}

// Tokenize an URL path using the slash separator ; the result does not have empty tokens
func tokenizePath(path string) []string {
	if "/" == path {
		return nil
	}
	return strings.Split(strings.Trim(path, "/"), "/")
}

// for debugging
func (r Route) String() string {
	return r.Method + " " + r.Path
}

// EnableContentEncoding (default=false) allows for GZIP or DEFLATE encoding of responses. Overrides the container.contentEncodingEnabled value.
func (r *Route) EnableContentEncoding(enabled bool) {
	r.contentEncodingEnabled = &enabled
}
